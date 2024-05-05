from __future__ import annotations
import argparse
import numpy as np
import pygame as pg 
from robots import CarLikeBot, CircleRobot
import gym
from random import uniform
# from env import Environment
from gym_env import Environment
from policies import CustomPolicy, CustomPolicyLessOl
from rewards import RewardCircle, MyReward, MyReward2
import time
import torch
import matplotlib.pyplot as plt

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

parser = argparse.ArgumentParser()

parser.add_argument("agent_type")
parser.add_argument("maps", type=list_of_ints)
parser.add_argument("policy")
parser.add_argument("pretrained")
parser.add_argument("reward", type=int)

args = parser.parse_args()

maps_ = args.maps
pretrained = args.pretrained

agent = None
if "car" in args.agent_type.lower():
    print("\nCARLIKE\n")
    agent = CarLikeBot
elif "circle" in args.agent_type.lower():
    agent = CircleRobot
else:
    raise Exception("agent_type must contain car or cicle")

"circke_89_rew1_lessol/model_160.pth" 
"carlike_gym/model_420.pth"
"carlike_gym/model_0.pth"
"circle_gym/model_140.pth"
"gym_models/model_1300.pth"
"my_Adam_my_plus/model_1800.pth"
"cont_SGD_my_600.pth"
"my_model_final.pth"

if args.reward == 0:
    rew = MyReward()
elif args.reward == 1:
    rew = MyReward2()

#CircleRobot  CarLikeBot
env = Environment(agent, map=maps_, reward=MyReward())
observation_space = gym.spaces.Box(low=np.zeros(16*4+3), high=np.ones(16*4+3))

pi = None
if "less" in args.policy.lower():
    pi = CustomPolicyLessOl(observation_space)
elif "std" in args.policy.lower():
    pi = CustomPolicy(observation_space)
else:
    raise Exception("policy must contain std or less")

if args.pretrained and  args.pretrained != "None": 
    pi.load_state_dict(torch.load(pretrained))

pg.init()
h, w = env.shape
surface = pg.display.set_mode((w, h))
font = pg.font.SysFont("arial", 15)
env.surface = surface
env.font = font

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
for i, re in enumerate(rew):
     x = np.arange(len(re))
     plt.plot(x, re, color=env.agents[i].color/max(env.agents[i].color), label=f'robot index {i}')

plt.legend()
plt.show()
