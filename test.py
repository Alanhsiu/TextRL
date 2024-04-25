#!/usr/bin/env python
# coding: utf-8

# # Controllable generation via RL to let Elon Musk speak ill of DOGE
# > How to control text generation through a sentiment classifier.
# 
# 

# In[ ]:


import torch
from datasets import load_from_disk
from vc.encodec_model.nar_bart_model import NARBartForConditionalGeneration
from transformers import (AutoTokenizer, BartForConditionalGeneration)
import pfrl

# define path
base_path = '/work/b0990106x/TextRL'
agent_input_dir = f'{base_path}/data-encodec'
agent_output_dir = f'{base_path}/output'
env_input_dir = agent_output_dir
env_output_dir = agent_input_dir

ar_checkpoint = "lca0503/speech-chatgpt-base-ar-v2-epoch10-wotrans"
nar_checkpoint = "lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans"

device = "cuda" if torch.cuda.is_available() else "cpu"
ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
ar_model = BartForConditionalGeneration.from_pretrained(ar_checkpoint)
nar_tokenizer = AutoTokenizer.from_pretrained(nar_checkpoint)
nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint)
ar_model.to(device)

dataset = load_from_disk(agent_input_dir)


# In[ ]:


all_src_encodec_layers = []
all_src_encodec = []
all_instruction = []
all_instruction_ids = []

data_len = len(dataset)
print(data_len)

# data_len = 30 # for testing
layer_len = 8

for i in range(layer_len):
    all_src_encodec_layers.append(dataset[f"src_encodec_{i}"])

for i in range(data_len):
    src_encodec = []
    for j in range(layer_len):        
        src_encodec.append(all_src_encodec_layers[j][i])
    all_src_encodec.append(src_encodec)

for i in range(data_len):
    all_instruction.append(dataset["instruction"][i])
    all_instruction_ids.append(ar_tokenizer(all_instruction[i])["input_ids"][1 : -1])


# In[ ]:


# import sys
# sys.path.append('/work/b0990106x/TextRL/vc')

from importlib import reload
import textrl
reload(textrl)

from textrl import TextRLEnv,TextRLActor
# reload(sys.modules['vc.trainer_encodec_vc_inference'])


# In[ ]:


from NISQA.nisqa.NISQA_model import nisqaModel

class MyRLEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish): # predicted will be the list of predicted token
        reward = 0
        
        if finish or len(predicted_list) >= self.env_max_length:
            try:
                # debug 0423
                print("Length of predicted_list:", len(predicted_list[0]))
                # if len(predicted_list[0]) < 300:
                #     return reward

                args_nisqa = {
                    'mode': 'predict_file', 
                    'pretrained_model': f'{base_path}/NISQA/weights/nisqa.tar', 
                    'deg': f'{base_path}/output/example.wav', 
                    'data_dir': None, 
                    'output_dir': f'{base_path}/NISQA/result',
                    'csv_file': None, 
                    'csv_deg': None,  
                    'num_workers': 0, 
                    'bs': 1,
                    'ms_channel': None
                }
                args_nisqa['tr_bs_val'] = args_nisqa['bs']
                args_nisqa['tr_num_workers'] = args_nisqa['num_workers']

                nisqa = nisqaModel(args_nisqa)
                prediction = nisqa.predict()
                reward = float(prediction['mos_pred'].iloc[0])
            except Exception as e:
                print("Error:", e)
                reward = 0

        return reward


# **fit one example**

# In[ ]:


#Debugging 
observation_list = []
# observation_list.append({'input': "", 'src_encodec': all_src_encodec[0], 'instruction': all_instruction[0]})
# observation_list.append({'input': "", 'src_encodec': all_src_encodec[0], 'instruction': all_instruction[0]})
# observation_list.append({'input': "", 'src_encodec': all_src_encodec[0], 'instruction': all_instruction[0]})
# observation_list.append({'input': "", 'src_encodec': all_src_encodec[0], 'instruction': all_instruction[0]})
# observation_list.append({'input': "", 'src_encodec': all_src_encodec[0], 'instruction': all_instruction[0]})
# observation_list.append({'input': "", 'src_encodec': all_src_encodec[0], 'instruction': all_instruction[0]})
# observation_list.append({'input': "", 'src_encodec': all_src_encodec[0], 'instruction': all_instruction[0]})
# observation_list.append({'input': "", 'src_encodec': all_src_encodec[0], 'instruction': all_instruction[0]})
# observation_list.append({'input': "", 'src_encodec': all_src_encodec[0], 'instruction': all_instruction[0]})
# observation_list.append({'input': "", 'src_encodec': all_src_encodec[0], 'instruction': all_instruction[0]})

for i in range(data_len):
    observation_list.append({'input': "", 'src_encodec': all_src_encodec[i], 'instruction': all_instruction[i]})


# In[ ]:


# for i in range(data_len):
#     print(f"Instruction {i}: ", observation_list[i]['instruction'])


# In[ ]:


env = MyRLEnv(ar_model, ar_tokenizer, nar_model, nar_tokenizer, observation_input=observation_list, compare_sample=1)
actor = TextRLActor(env, ar_model, ar_tokenizer)
agent = actor.agent_ppo(update_interval=10, minibatch_size=2000, epochs=1)


# In[ ]:


import logging
import os
import sys

output_log_path = 'log/log.log'
output_file_path = 'log/output.txt'

if not os.path.exists('log'):
    os.makedirs('log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handlers = logger.handlers[:]
for handler in handlers:
    if isinstance(handler, logging.StreamHandler):
        logger.removeHandler(handler)
    
file_handler = logging.FileHandler(output_log_path)
logger.addHandler(file_handler)

class StreamToLogger(object):
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

# print("This will go to the log file.")
# logger.info("This is an info message.")
# logger.error("This error will also be logged to the file.")


# In[ ]:


import sys
import time

start_time = time.time()

with open(output_file_path, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f

    pfrl_outdir = 'train-0424'
    # pfrl.experiments.train_agent_with_evaluation(
    #     agent,
    #     env,
    #     steps=500000,
    #     eval_n_steps=None, 
    #     eval_n_episodes=3, 
    #     train_max_episode_len=1000,  
    #     eval_interval=1000,
    #     outdir=pfrl_outdir, 
    #     logger=logger,
    #     use_tensorboard=True,
    #     checkpoint_freq=50000
    # )
    pfrl.experiments.train_agent_with_evaluation(
        agent,
        env,
        steps=100000,
        eval_n_steps=None, 
        eval_n_episodes=3, 
        train_max_episode_len=1000,  
        eval_interval=1000,
        outdir=pfrl_outdir, 
        logger=logger,
        use_tensorboard=True,
        checkpoint_freq=10000
    )
    
    
    print('Output has been written to', output_file_path)
    print("used time: ", time.time() - start_time)
    sys.stdout = original_stdout


# loading the best result and predict.

# 

# In[ ]:


# agent.load(pfrl_outdir + '/best')


# In[ ]:


# actor.predict(observation_list[0])

