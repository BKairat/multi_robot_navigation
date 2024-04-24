import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import numpy as np
import pygame as pg
from robots import CircleRobot, CarLikeBot
from robots import CarLikeBot, CircleRobot
from history import History, MyHistory
from rewards import MyReward
import matplotlib.pyplot as plt
from gym_env import Environment

# Create the environment
# env = gym.make('CartPole-v1')

# Create the PPO agent
# model = PPO('MlpPolicy', env, verbose=1)

# Train the agent for 10000 steps
# model.learn(total_timesteps=10000)

# # Save the trained model
# model.save("ppo_cartpole")
# print(env.action_space.n)
# Evaluate the trained agent
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward}")

map_ = [5,6,7]

env = Environment(CircleRobot, MyReward, map=map_)
pg.init()
h, w = env.shape
surface = pg.display.set_mode((w, h))
font = pg.font.SysFont("arial", 15)
env.surface = surface
env.font = font

while running:
    surface.fill("white")
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    actions = []
    env.step(actions)

    env.draw()
    pg.display.flip()
    # time.sleep(5)
    if a >= 512 :
        # break
        env.reset()
        a = 0
    else:
        a += 1

pg.quit()