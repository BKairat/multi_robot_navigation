from __future__ import annotations
import numpy as np
import pygame as pg 
from robots import CarLikeBot, CircleRobot
from random import uniform
from env import Environment
from policies import ContiniousPolicy, DiscterePolicy, ContiniousPolicy001, DiscterePolicy001
from rewards import RewardCircle, MyReward
import time
import torch
import matplotlib.pyplot as plt

disc = 1
maps_ = [8, 9]
pretrained = ""
"my_model_final.pth"
my = True


env = Environment(CircleRobot, int(np.random.choice(maps_)), reward=MyReward(), discrete=disc)
pg.init()
h, w = env.shape
surface = pg.display.set_mode((w, h))
font = pg.font.SysFont("arial", 15)
env.surface = surface
env.font = font

if disc:
    p = DiscterePolicy001()
    if pretrained:
        p.load_state_dict(torch.load(pretrained))
else:
    if my:
        p = ContiniousPolicy001()
        if pretrained:
            p.load_state_dict(torch.load(pretrained))
    else:
        p = ContiniousPolicy()
        if pretrained:
            p.load_state_dict(torch.load(pretrained))
clock = pg.time.Clock()
running = True
a = 0

rew = [[] for _ in env.agents]

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
    ol, op = [], []
    for ag in env.agents:
        ol_, op_ = ag.history.get_vectors()
        ol.append(ol_)
        op.append(op_)
    
    ol = torch.tensor(ol).float()
    op = torch.tensor(op).float()

    actions = p.sample_actions(ol, op)
    
    # print(actions)
    r, done = env.step(actions)
    for r_i in range(len(r)):
        rew[r_i].append(r[r_i])
        
    env.draw()
    pg.display.flip()
    # time.sleep(0.0)
    if a >= 512 or done:
        # break
        env.reset(int(np.random.choice(maps_)))
        # if disc:
        #     p = DiscterePolicy()
        # env.get_observation()
        a = 0
        
    else:
        a += 1

    clock.tick(500000)

pg.quit()
for i, re in enumerate(rew):
    x = np.arange(len(re))
    plt.plot(x, re, color=env.agents[i].color/max(env.agents[i].color), label=f'robot index {i}')

plt.legend()
plt.show()