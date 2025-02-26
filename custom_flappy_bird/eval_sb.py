"""Simple script to evaluate on a sb3 saved policy"""

import pathlib
import os
import glob

# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
from utils.utils import get_time_str, load_config
# from utils.sb3_callbacks import FlapActionMetricCallback
from utils.wrappers import RecordBestVideo

models_dir = pathlib.Path(__file__).parent.parent.resolve().joinpath('models')
run = 'PPO_20250225-210152'

run_folder = os.path.join(models_dir, run)
model = os.path.join(run_folder, 'best_model.zip')

config_path = glob.glob(os.path.join(run_folder, '*.json'))[0]
config = load_config(config_path)

num_cpu = 10
n_eval_episodes = 20

register(
     id="CustomFlappyBirdEnv",
     entry_point="gym_env.custom_flappy_env:CustomFlappyBirdEnv",
)

vec_env = make_vec_env(
    "CustomFlappyBirdEnv",
    n_envs=num_cpu,
    env_kwargs=config['env_kwargs'],
    monitor_dir='./monitor',
    wrapper_class=RecordBestVideo,
    wrapper_kwargs={
          'video_folder': f'./replays/eval/run_{get_time_str()}',
          'name_prefix': "sb3-flappy",
          'record_mode': "best",
          'reward_in_name': True,
          'second_metric': 'score',
    }
    )
alg = PPO.load(model, env=vec_env, device='cpu')

mean_reward, std_reward = evaluate_policy(
    alg,
    env=vec_env,
    n_eval_episodes=n_eval_episodes,
    )

print('Mean reward: ', mean_reward)
print('Std reward: ', std_reward)
