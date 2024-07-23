'''
Sample implementation of an RL problem using Stable Baseline 3
'''

# Import libraries
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

# Create environment (vector environment)
env_name = 'CartPole-v1'
num_train_envs = 4
num_eval_envs= 5

# log dir to save eval results
eval_log_dir = './eval_logs/'
os.makedirs(eval_log_dir, exist_ok = True)

# Video record
video_folder = 'logs/video/'
video_length = 100

vec_env = make_vec_env(env_name, num_train_envs, seed = 42)
# Use eval env callback
eval_env = make_vec_env(env_name, num_eval_envs, seed = 42)

# Callback
eval_callback = EvalCallback(eval_env, best_model_save_path = eval_log_dir, eval_freq = max(500 // num_train_envs, 1), n_eval_episodes = 5, deterministic = True, render = False)
# Instantiate agent
model = DQN('MlpPolicy', vec_env, verbose = 1) # MlpPolicy for vector based and CnnPolicy for frame based inputs ; use default hyperparameters

# Train
# model.learn(total_timesteps = int(2e3), progress_bar = True)
model.learn(total_timesteps = int(2e3), progress_bar = True, callback = eval_callback)

# Save the agent
model.save(f'{env_name}')

del model

# Load the agent
model = DQN.load(f'{env_name}', env = vec_env)

# Evaluate agent (normal)
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes = 10)

# Test agent

# Record the video starting at the first step
vec_env = VecVideoRecorder(vec_env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"random-agent-{env_name}")

# vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic = True)
    obs, rewards, dones, info = vec_env.step(action)

    vec_env.render('human')
    
# Save the video
vec_env.close()