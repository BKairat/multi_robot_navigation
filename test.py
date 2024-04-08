from __future__ import annotations
import numpy as np
import pygame as pg 
from robots import CarLikeBot, CircleRobot
from random import uniform
from env import Environment
from policies import ContiniousPolicy, DiscterePolicy
from rewards import RewardCircle
import time
import torch
import matplotlib.pyplot as plt

disc = True
maps_ = [4,5,6,7]


env = Environment(CircleRobot, int(np.random.choice(maps_)), reward=RewardCircle(), discrete=disc)
pg.init()
h, w = env.shape
surface = pg.display.set_mode((w, h))
font = pg.font.SysFont("arial", 15)
env.surface = surface
env.font = font
env.get_observation()

if disc:
    p = DiscterePolicy()
    # p.load_state_dict(torch.load('model_disc_50000.pth'))
else: 
    p = ContiniousPolicy()
    p.load_state_dict(torch.load('pretrained_models/model_with_one_tanh.pth'))
clock = pg.time.Clock()
running = True
a = 0

rew = []

while running:
    surface.fill("white")
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    env.get_observation()

    ol, od, og, ov = [], [], [], []
    for ag in env.agents:
        ol_, od_, og_, ov_ = ag.history.get_vectors()
        ol.append(ol_)
        od.append(od_)
        og.append(og_)
        ov.append(ov_)
    
    ol = torch.tensor(ol).float()
    od = torch.tensor(od).float()
    og = torch.tensor(og).float()
    ov = torch.tensor(ov).float()

    actions = p.sample_actions(ol, od, og, ov)
    
    r, done = env.step(actions)
    rew.append(r[0])
    env.draw()
    pg.display.flip()
    time.sleep(0.05)
    if a >= 100 or done:
        env.reset(int(np.random.choice(maps_)))
        if disc:
            p = DiscterePolicy()
        env.get_observation()
        a = 0
        break
    else:
        a += 1

    clock.tick(500000)

pg.quit()
x = np.arange(len(rew))
plt.plot(x, rew)
plt.show()