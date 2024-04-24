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
maps_ = [8]
pretrained = "gym_models/model_1300.pth"
"my_Adam_my_plus/model_1800.pth"
# pretrained = "my_Adam_my_plus_1800.pth"
"cont_SGD_my_600.pth"

"my_model_final.pth"
my = True

env = Environment(CircleRobot, map=[8, 9], reward=MyReward())
observation_space = gym.spaces.Box(low=np.zeros(16*4+3), high=np.ones(16*4+3))

pi = CustomPolicy(observation_space)
# env = Environment(CircleRobot, int(np.random.choice(maps_)), reward=MyReward(), discrete=disc, observation_=True)
pg.init()
h, w = env.shape
surface = pg.display.set_mode((w, h))
font = pg.font.SysFont("arial", 15)
env.surface = surface
env.font = font

pi.load_state_dict(torch.load(pretrained))

# if disc:
#     p = DiscterePolicy001()
#     if pretrained:
#         p.load_state_dict(torch.load(pretrained))
# else:
#     if my:
#         p = ContiniousPolicy001()
#         if pretrained:
#             p.load_state_dict(torch.load(pretrained))
#     else:
#         p = ContiniousPolicy()
#         if pretrained:
#             p.load_state_dict(torch.load(pretrained))
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

    # env.get_observation()

    # ol, od, og, ov = [], [], [], []
    # for ag in env.agents:
    #     ol_, od_, og_, ov_ = ag.history.get_vectors()
    #     ol.append(ol_)
    #     od.append(od_)
    #     og.append(og_)
    #     ov.append(ov_)
    
    # ol = torch.tensor(ol).float()
    # od = torch.tensor(od).float()
    # og = torch.tensor(og).float()
    # ov = torch.tensor(ov).float()

    # actions = p.sample_actions(ol, od, og, ov)
    # ol, op = [], []
    # for ag in env.agents:
    #     ol_, op_ = ag.history.get_vectors()
    #     ol.append(ol_)
    #     op.append(op_)
    
    # ol = torch.tensor(ol).float()
    # op = torch.tensor(op).float()

    actions = pi.sample_actions(torch.tensor(s).float())
    print(actions)
    # actions = p.sample_actions(ol, op)
    # print(actions[:, 1].item())

    # print(actions)
    s, r, done, _ = env.step(np.array(actions))
    for r_i in range(len(r)):
        rew[r_i].append(r[r_i])
        
    env.render()
    pg.display.flip()
    # time.sleep(5)
    if a >= 512 or done.all():
        # break
        s = env.reset()
        
        # if disc:
        #     p = DiscterePolicy()
        # env.get_observation()
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