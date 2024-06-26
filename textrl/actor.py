import itertools

import numpy as np
import pfrl
import torch
import torch.nn.functional as F
from pfrl.agents.ppo import _elementwise_clip
from pfrl.utils.mode_of_distribution import mode_of_distribution
from torch import autocast
from pfrl.utils.batch_states import batch_states
from pfrl.utils.mode_of_distribution import mode_of_distribution
from pfrl.utils.recurrent import (
    concatenate_recurrent_states,
    flatten_sequences_time_first,
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
    pack_and_forward,
)


def get_modulelist_pos(model):
    module_list_pos = 0
    for ids, i in enumerate(list(model.children())):
        if isinstance(i, torch.nn.ModuleList):
            module_list_pos = ids
    return module_list_pos


class HFModelListModule(torch.nn.Module):
    def __init__(self, module_list):
        super(HFModelListModule, self).__init__()
        self.module_list = module_list

    def forward(self, hidden):
        for module in self.module_list:
            hidden = module(hidden)[0]
        return hidden


class TextRLActor:
    def __init__(self, env, model, tokenizer, optimizer='sgd', gpu_id=0,
                 unfreeze_layer_from_past=0,
                #  act_deterministically=True,
                 act_deterministically=False,
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0):
        self.agent = None
        self.n_actions = max(model.config.vocab_size, tokenizer.vocab_size)
        self.env = env
        self.gpu_id = gpu_id
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.model = model
        if hasattr(model.config, 'word_embed_proj_dim'):
            self.obs_size = model.config.word_embed_proj_dim
        else:
            self.obs_size = model.config.hidden_size
        self.converter = self.model.lm_head
        self.act_deterministically = act_deterministically
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.optimizer = optimizer
        self.unfreeze_layer_from_past = unfreeze_layer_from_past

        parents = [parent[0] for parent in model.named_children()]
        if 'transformer' in parents:  # gpt2/bloom:
            transformers_model = model.transformer
        elif 'model' in parents:  # bart
            transformers_model = model.model
        elif 'decoder' in parents:  # t5
            transformers_model = model.decoder
        else:
            raise ValueError('model not supported')

        if unfreeze_layer_from_past > 0:
            self.middle_model = HFModelListModule(list(transformers_model.children())
                                                  [get_modulelist_pos(transformers_model)]
                                                  [-self.unfreeze_layer_from_past:])
            self.remaining_model = torch.nn.Sequential(
                *list(transformers_model.children())[get_modulelist_pos(transformers_model) + 1:])
        else:
            self.middle_model = torch.nn.Sequential()
            self.remaining_model = torch.nn.Sequential()

    def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20, lr=3e-6, entropy_coef=0.1):
        policy = torch.nn.Sequential(
            self.middle_model,
            self.remaining_model,
            self.converter,
            SoftmaxCategoricalHead(self.env,
                                   temperature=self.temperature,
                                   top_k=self.top_k,
                                   top_p=self.top_p)
        )
        vf = torch.nn.Sequential(
            torch.nn.Linear(self.obs_size, self.obs_size // 2),
            torch.nn.Linear(self.obs_size // 2, self.obs_size // 4),
            torch.nn.Linear(self.obs_size // 4, 1)
        )
        model = pfrl.nn.Branched(policy, vf)
        if isinstance(self.optimizer, str):
            if self.optimizer.lower() == 'adamw':
                opt = torch.optim.AdamW(model.parameters(), lr=lr)
            else:
                opt = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            opt = self.optimizer
        model = model.cuda()
        agent = TextPPO(
            model,
            opt,
            gpu=self.gpu_id,
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            clip_eps_vf=None,
            # entropy_coef=0,
            entropy_coef=entropy_coef,
            gamma=0.95,  # https://arxiv.org/abs/2210.01241
            lambd=1,
            max_grad_norm=1.0,
            standardize_advantages=True,
            act_deterministically=self.act_deterministically
        )
        self.agent = agent
        return agent

    @autocast('cuda')
    def predict(self, input_item):
        t = 0
        with torch.inference_mode():
            with self.agent.eval_mode():
                obs = self.env.reset(input_item)
                while True:
                    action = self.agent.act(obs)
                    obs, reward, done, pred = self.env.step(action)
                    t += 1
                    reset = t >= self.env.env_max_length
                    self.agent.observe(obs, reward, done, reset)
                    if done or reset:
                        return pred.get('predicted_str')
                    


def top_k_top_p_filtering(
        logits: torch.FloatTensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_p = float(top_p)
    if top_k > 0:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if 0 < top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


class SoftmaxCategoricalHead(torch.nn.Module):
    def __init__(self, env, temperature=1.0, top_k=0, top_p=1.0):
        super().__init__()
        self.env = env
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def forward(self, logits):
        logits = logits / self.temperature
        logits = top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)
        return torch.distributions.Categorical(logits=logits)


import os
import pickle
from datetime import datetime

class TextPPO(pfrl.agents.PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_count = 0
        self.current_time = datetime.now().strftime("%m%d-%H%M")
        self.pickle_folder_name = f"replay_buffer/{self.current_time}"
    ##### saving replay buffer (0425)
    def dump_replay_buffer(self, filename="replay_buffer.pkl"):
        print("Dumping replay buffer")
        directory = self.pickle_folder_name
        if not os.path.exists(directory):
            os.makedirs(directory)
        output_filename = os.path.join(directory, filename)
        with open(output_filename, "wb") as f:
            pickle.dump(self.memory, f)
    ##### end of saving replay buffer

    def _flush_last_episode(self):
        if self.last_episode:
            self.memory.append(self.last_episode)
            self.last_episode = []
        if self.batch_last_episode:
            for i, episode in enumerate(self.batch_last_episode):
                if episode:
                    datasetsize = (
                            sum(len(episode) for episode in self.memory)
                            + len(self.last_episode)
                            + (
                                0
                                if self.batch_last_episode is None
                                else sum(len(episode) for episode in self.batch_last_episode)
                            )
                    )
                    # print(f"FLUSH: datasetsize: {datasetsize}")
                    # print(f"FLUSH: Length of self.batch_last_episode[0] {i}: {len(self.batch_last_episode[0])}")
                    # print(f"FLUSH: Length of episode {i}: {len(episode)}")
                    # print(f"FLUSH: Length of self.memory BEFORE append {i}: {len(self.memory)}")
                    self.memory.append(episode)
                    self.batch_last_episode[i] = []
                    
        #     for i, saved_episode in enumerate(self.memory):
        #         print(f"FLUSH: Length of self.memory[{i}]: {len(saved_episode)}")
        # print("Done Flushing")

    def _update_if_dataset_is_ready(self):   
        dataset_size = (
                sum(len(episode) for episode in self.memory)
                + len(self.last_episode)
                + (
                    0
                    if self.batch_last_episode is None
                    else sum(len(episode) for episode in self.batch_last_episode)
                )
        )
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            if self.recurrent:
                dataset = pfrl.agents.ppo._make_dataset_recurrent(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                    device=self.device,
                )
                self._update_recurrent(dataset)
            else:
                dataset = pfrl.agents.ppo._make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    device=self.device,
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
                print("Update Policy and Value Function")
                
            ##### added for saving replay buffer (0425)
            # Uncomment the following line to save replay buffer after each update
            # self.dump_replay_buffer(f"replay_buffer_update_{self.update_count}.pkl")
            # self.update_count += 1
            ##### end of added for saving replay buffer  
            
            self.explained_variance = self._compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory))
            )
            self.memory = []


    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert self.training

        for i, (state, action, reward, next_state, done, reset) in enumerate(
            zip(
                self.batch_last_state,
                self.batch_last_action,
                batch_reward,
                batch_obs,
                batch_done,
                batch_reset,
            )
        ):
            if state is not None:
                assert action is not None
                transition = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "nonterminal": 0.0 if done else 1.0,
                }
                if self.recurrent:
                    transition["recurrent_state"] = get_recurrent_state_at(
                        self.train_prev_recurrent_states, i, detach=True
                    )
                    transition["next_recurrent_state"] = get_recurrent_state_at(
                        self.train_recurrent_states, i, detach=True
                    )
                self.batch_last_episode[i].append(transition)
                # print(f"Step Reward: {reward}")
            if done or reset:
                # print(f"Step Reward on Done Step: {reward}")
                assert self.batch_last_episode[i]
                self.memory.append(self.batch_last_episode[i])
                # print("Append to memory a size of ", len(self.batch_last_episode[i]))
                # for j, saved_episode in enumerate(self.memory):
                #     print(f"Length of self.memory[{j}] now: {len(saved_episode)}")
                self.batch_last_episode[i] = []
            self.batch_last_state[i] = None
            self.batch_last_action[i] = None

        self.train_prev_recurrent_states = None
        # if self.recurrent:
        #     # Reset recurrent states when episodes end
        #     indices_that_ended = [
        #         i
        #         for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
        #         if done or reset
        #     ]
        #     if indices_that_ended:
        #         self.train_recurrent_states = mask_recurrent_state_at(
        #             self.train_recurrent_states, indices_that_ended
        #         )

        if all(batch_done):
            print("All batch_done, episode done and update if ready")
            self._update_if_dataset_is_ready()
        # self._update_if_dataset_is_ready()


    def _compute_explained_variance(self, transitions):
        """Compute 1 - Var[return - v]/Var[return].

        This function computes the fraction of variance that value predictions can
        explain about returns.
        """
        t = np.array([tr["v_teacher"] for tr in transitions])
        y = np.array([tr["v_pred"] for tr in transitions])
        vart = np.var(t)
        if vart == 0:
            return np.nan
        else:
            return float(1 - np.var(np.average(t) - y) / vart)

    def batch_act(self, batch_obs):
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    @autocast('cuda')
    def _batch_act_eval(self, batch_obs):
        assert not self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            action_distrib, _ = self.model(b_state)
            if self.act_deterministically:
                action = mode_of_distribution(action_distrib).cpu().numpy()
            else:
                action = action_distrib.sample().cpu().numpy()

        return action

    def _lossfun(
            self, entropy, vs_pred, log_probs, vs_pred_old, log_probs_old, advs, vs_teacher
    ):
        prob_ratio = torch.exp(log_probs - log_probs_old)
        loss_policy = -torch.mean(
            torch.min(
                (prob_ratio * advs),
                torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs,
            ),
        )
        if self.clip_eps_vf is None:
            loss_value_func = F.mse_loss(vs_pred.squeeze(), vs_teacher.squeeze())
        else:
            clipped_vs_pred = _elementwise_clip(
                vs_pred,
                vs_pred_old - self.clip_eps_vf,
                vs_pred_old + self.clip_eps_vf,
            )
            loss_value_func = torch.mean(
                torch.max(
                    F.mse_loss(vs_pred.squeeze(), vs_teacher, reduction="none"),
                    F.mse_loss(clipped_vs_pred.squeeze(), vs_teacher, reduction="none"),
                )
            )
        loss_entropy = -torch.mean(entropy)

        self.value_loss_record.append(float(loss_value_func))
        self.policy_loss_record.append(float(loss_policy))
        loss = (
                loss_policy
                + self.value_func_coef * loss_value_func
                + self.entropy_coef * loss_entropy
        )
        return loss
