from __future__ import annotations
import numpy as np
import pygame as pg 
from robots import CarLikeBot, CircleRobot
from random import uniform
from env import Environment
from policies import ContiniousPolicy, DiscterePolicy, ContiniousPolicy001
from rewards import RewardCircle, MyReward
import time
import torch
from train import PPO, MyPPO

disc = 0
maps_ = [8, 9]

env = Environment(CircleRobot, int(np.random.choice(maps_)), reward=MyReward(), discrete=disc)
if disc:
    p = DiscterePolicy()
    p.load_state_dict(torch.load('model_disc.pth'))
else: 
    p = ContiniousPolicy001()
    # p.load_state_dict(torch.load('pretrained_models/model_with_one_tanh.pth'))
    
MyPPO(
    env,
    p,
    path = None,
    max_steps = 100,
    epochs = 10_001,
    lr = 0.0003,
    gamma=0.95,
    epsilon = 0.2,
    sgd_iters = 4,
    maps = maps_,
    test_feq = 5_00,
    test_num = 25,
    model_name = "my_model"
    )