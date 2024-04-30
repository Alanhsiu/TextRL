#!/usr/bin/env python
# coding: utf-8

# # Controllable generation via RL about text-guided voice conversion
# 

# In[ ]:


import torch
from datasets import load_from_disk
from vc.encodec_model.nar_bart_model import NARBartForConditionalGeneration
from transformers import AutoTokenizer, BartForConditionalGeneration

# load the model
ar_checkpoint = "lca0503/speech-chatgpt-base-ar-v2-epoch10-wotrans"
nar_checkpoint = "lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans"

device = "cuda" if torch.cuda.is_available() else "cpu"
ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
ar_model = BartForConditionalGeneration.from_pretrained(ar_checkpoint)
nar_tokenizer = AutoTokenizer.from_pretrained(nar_checkpoint)
nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint)
ar_model.to(device)


# In[ ]:


from datetime import datetime
import os

now = datetime.now()
ts = now.strftime("%m%d-%H%M")
print("timestamp:", ts)

# define the path
base_path = "/work/b0990106x/TextRL"
agent_input_dir = f"{base_path}/data-encodec"
agent_output_dir = f"{base_path}/output/{ts}"
env_input_dir = agent_output_dir
env_output_dir = agent_input_dir

if not os.path.exists(agent_output_dir):
    os.makedirs(agent_output_dir)


# In[ ]:


# load the dataset
dataset = load_from_disk(agent_input_dir)

all_src_encodec_layers = []
all_src_encodec = []
all_instruction = []
# all_instruction_ids = []

layer_len = 8
data_len = 3
# data_len = len(dataset)
print("data_len:", data_len)

for i in range(layer_len):
    all_src_encodec_layers.append(dataset[f"src_encodec_{i}"])

for i in range(data_len):
    src_encodec = []
    for j in range(layer_len):
        src_encodec.append(all_src_encodec_layers[j][i])
    all_src_encodec.append(src_encodec)

    all_instruction.append(dataset["instruction"][i])
    # all_instruction_ids.append(ar_tokenizer(all_instruction[i])["input_ids"][1 : -1])


# In[ ]:


from importlib import reload
import textrl

reload(textrl)

from textrl import TextRLEnv, TextRLActor
from NISQA.nisqa.NISQA_model import nisqaModel


class MyRLEnv(TextRLEnv):
    def get_reward(self, _, predicted_list, finish):
        reward = 0
        if finish or len(predicted_list) >= self.env_max_length:
            try:
                args_nisqa = {
                    "mode": "predict_file",
                    "pretrained_model": f"{base_path}/NISQA/weights/nisqa.tar",
                    "deg": f"{base_path}/output/{ts}/example.wav",
                    "data_dir": None,
                    "output_dir": f"{base_path}/NISQA/result/",
                    "csv_file": None,
                    "csv_deg": None,
                    "num_workers": 0,
                    "bs": 1,
                    "ms_channel": None,
                }
                args_nisqa["tr_bs_val"] = args_nisqa["bs"]
                args_nisqa["tr_num_workers"] = args_nisqa["num_workers"]

                nisqa = nisqaModel(args_nisqa)
                prediction = nisqa.predict()
                reward = float(prediction["mos_pred"].iloc[0])
                print(
                    "Length of predicted_list:",
                    len(predicted_list[0]),
                    ", Reward:",
                    reward,
                )

            except Exception as e:
                print("Error:", e)
                reward = 0

        return reward


# In[ ]:


observation_list = []
for i in range(3):
    observation_list.append(
        {
            "input": "",
            "src_encodec": all_src_encodec[i],
            "instruction": all_instruction[i],
        }
    )
    print("src_encodec:", observation_list[i]["src_encodec"][0])
    print("instruction:", all_instruction[i])

# for i in range(data_len):
#     observation_list.append({'input': "", 'src_encodec': all_src_encodec[i], 'instruction': all_instruction[i]})


# In[ ]:


print("observation_list:", observation_list)


# In[ ]:


from types import SimpleNamespace

args_predict = SimpleNamespace(
    output_path=f"{base_path}/output/{ts}/example.wav", seed=0, device="cuda"
)

env = MyRLEnv(
    ar_model,
    ar_tokenizer,
    nar_model,
    nar_tokenizer,
    args_predict,
    observation_input=observation_list,
    compare_sample=1,
)
actor = TextRLActor(env, ar_model, ar_tokenizer, ts)
agent = actor.agent_ppo(update_interval=1500, minibatch_size=2000, epochs=1)


# In[ ]:


import logging
import os
import sys

output_log_path = f"log/log_{ts}.log"
output_file_path = f"log/output_{ts}.txt"

if not os.path.exists("log"):
    os.makedirs("log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handlers = logger.handlers[:]
for handler in handlers:
    logger.removeHandler(handler)

file_handler = logging.FileHandler(output_log_path)
logger.addHandler(file_handler)


# class StreamToLogger(object):
#     def __init__(self, logger, log_level):
#         self.logger = logger
#         self.log_level = log_level
#         self.linebuf = ""

#     def write(self, buf):
#         for line in buf.rstrip().splitlines():
#             self.logger.log(self.log_level, line.rstrip())

#     def flush(self):
#         pass


# sys.stdout = StreamToLogger(logger, logging.INFO)
# sys.stderr = StreamToLogger(logger, logging.ERROR)


# In[ ]:


import sys
import time
import pfrl

start_time = time.time()

with open(output_file_path, "w") as f:
    original_stdout = sys.stdout
    sys.stdout = f

    pfrl_outdir = f"ckpt/train_{ts}"

    pfrl.experiments.train_agent_with_evaluation(
        agent,
        env,
        steps=3000,
        eval_n_steps=None,
        eval_n_episodes=3,
        train_max_episode_len=1000,
        eval_interval=1000,
        outdir=pfrl_outdir,
        logger=logger,
        use_tensorboard=True,
        checkpoint_freq=1000,
    )

    print("Output has been written to", output_file_path)
    print("used time: ", time.time() - start_time)
    sys.stdout = original_stdout


# In[ ]:


agent.load(pfrl_outdir + "/best")
actor.predict(observation_list[0])


# In[ ]:


print(ts)


# In[ ]:


import read_pickle
# ts = "0430-0955"
for i in range(10):
    pickle_file = f"replay_buffer_update_{i}.pkl"
    pickle_data = read_pickle.load_pickle(file_path = f'{base_path}/replay_buffer/{ts}/{pickle_file}')
    length_of_pickle_data = len(pickle_data[0])
    print(f"length of {pickle_file}: {length_of_pickle_data}") 
 

