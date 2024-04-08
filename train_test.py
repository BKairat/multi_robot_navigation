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
from train import PPO

disc = True
maps_ = [4,5,6,7]

env = Environment(CircleRobot, int(np.random.choice(maps_)), reward=RewardCircle(), discrete=disc)
if disc:
    p = DiscterePolicy()
    p.load_state_dict(torch.load('model_disc.pth'))
else: 
    p = ContiniousPolicy()
    p.load_state_dict(torch.load('pretrained_models/model_with_one_tanh.pth'))
    
PPO(
    env,
    p,
    path = None,
    max_steps = 128,
    epochs = 15_001,
    lr = 0.0003,
    gamma=0.95,
    epsilon = 0.2,
    sgd_iters = 8,
    maps = maps_,
    test_feq = 5_00,
    test_num = 25
    )