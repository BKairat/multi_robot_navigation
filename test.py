from __future__ import annotations
import numpy as np
import pygame as pg 
from robots import CarLikeBot, CircleRobot
import gym
from random import uniform
# from env import Environment
from gym_env import Environment
# from policies import ContiniousPolicy, DiscterePolicy, ContiniousPolicy001, DiscterePolicy001, testPolicy
from policies import CustomPolicy
from rewards import RewardCircle, MyReward
import time
import torch
import matplotlib.pyplot as plt

disc = 0
maps_ = [4,5,6,7]
pretrained = "carlike_gym/model_520.pth"
"carlike_gym/model_0.pth"
"circle_gym/model_140.pth"
"gym_models/model_1300.pth"
"my_Adam_my_plus/model_1800.pth"
"cont_SGD_my_600.pth"

"my_model_final.pth"
my = True
# CircleRobot  CarLikeBot
env = Environment(CarLikeBot, map=maps_, reward=MyReward())
observation_space = gym.spaces.Box(low=np.zeros(16*4+3), high=np.ones(16*4+3))

pi = CustomPolicy(observation_space)

pg.init()
h, w = env.shape
surface = pg.display.set_mode((w, h))
font = pg.font.SysFont("arial", 15)
env.surface = surface
env.font = font

pi.load_state_dict(torch.load(pretrained))

clock = pg.time.Clock()
running = True
a = 0

rew = [[] for _ in env.agents]

s = env.reset()
while running:
    surface.fill("white")
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    actions = pi.sample_actions(torch.tensor(s).float())
    print(actions)

    s, r, done, _ = env.step(np.array(actions))
    for r_i in range(len(r)):
        rew[r_i].append(r[r_i])
        
    env.render()
    pg.display.flip()

    if a >= 126 or done.all():
        s = env.reset()
        a = 0
    else:
        a += 1

    clock.tick(500000)

pg.quit()
# for i, re in enumerate(rew):
#     x = np.arange(len(re))
#     plt.plot(x, re, color=env.agents[i].color/max(env.agents[i].color), label=f'robot index {i}')

# plt.legend()
# plt.show()
