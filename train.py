from __future__ import annotations
import argparse
import numpy as np
from robots import CarLikeBot, CircleRobot
# from random import uniform
from gym_env import Environment
from policies import CustomPolicy, CustomPolicyLessOl 
# from algo.ppo import PPO
from rewards import RewardCircle, MyReward, MyReward2
import gym
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import os

        
class PPO_gym:
        def __init__(
                self,
                train_env,
                policy,
                n_epochs = 100,
                gamma = 0.99,
                lr: float = 3e-4,
                max_steps: int = 126,
                vf_coef: float = 0.5,
                max_grad_norm: float = 0.5,
                gae_lambda: float = 0.95,
                clip_range: float = 0.2,
                ent_coef: float = 0.0,
                freq: int = 20,
                path: str = "test"
                ):
                self.train_env = train_env
                self.policy = policy
                self.n_epochs = n_epochs
                self.clip_range = clip_range
                self.lr = lr
                self.gamma = gamma
                self.vf_coef = vf_coef
                self.max_grad_norm = max_grad_norm
                self.gae_lambda = gae_lambda
                self.max_steps = max_steps
                self.ent_coef = ent_coef
                self.freq = freq
                self.path = path

        def tarin(self):
                optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

                mean_rewards, p_losses, v_losses = np.zeros(self.n_epochs), np.zeros(self.n_epochs), np.zeros(self.n_epochs)  # for logging mean rewards over epochs
                for epoch in range(self.n_epochs):
                        tensor_s, tensor_a, tensor_r = self.sample_trajectories()  # collect trajectories using current policy
                        tensor_r = tensor_r.transpose(0,1)
                        # print(tensor_s.shape, tensor_a.shape, tensor_r.shape)
                        with torch.no_grad():  # compute the old probabilities
                                logp_old = self.policy.log_prob(tensor_a, tensor_s[:self.max_steps])  # compute log(pi(a_t | s_t))
                                # print("log_prob.shape", logp_old.shape, "\n\n")
                        for _ in range(6):  # we can even do multiple gradient steps
                                values = self.policy.value_estimates(tensor_s).squeeze(dim=-1)  # estimate value function for all states 
                                # print("values.shape", values.shape, "\n\n")
                                logp = self.policy.log_prob(tensor_a, tensor_s[:self.max_steps])  # compute log(pi(a_t | s_t))
                                # print("logp.shape", logp.shape, "\n\n")
                                with torch.no_grad():  # no need for gradients when computing the advantages and value targets
                                        # value_targets, advantage_estimates = compute_advantage_estimates(tensor_r, values, gamma, bootstrap=True)
                                        value_targets, advantage_estimates = self.compute_gae(tensor_r, values, self.gamma, lambda_=self.gae_lambda)
                                        advantage_estimates = (advantage_estimates - advantage_estimates.mean()) / advantage_estimates.std()  # normalize advantages
                                # print("value_targets.shape", value_targets.shape, "\n\n")
                                # print("advantage_estimates.shape", advantage_estimates.shape, "\n\n")

                                L_v = F.mse_loss(value_targets, values[:, :-1])  # add the value loss
                                L_e = self.entropy_loss(logp)
                                
                                p_ratios = torch.exp(logp - logp_old)  # compute the ratios r_\theta(a_t | s_t)
                                L_ppo = self.ppo_loss(advantage_estimates, p_ratios)  # compute the policy gradient loss
                                
                                total_loss = self.vf_coef*L_v + L_ppo + self.ent_coef*L_e
                                
                                optim.zero_grad()
                                total_loss.backward()
                                optim.step()
                                
                        if epoch % self.freq == 0:
                                print('Epoch %d, mean reward: %.3f, value loss: %.3f, ppo_loss: %.6f' % (epoch, tensor_r.mean(), L_v.item(), L_ppo.item()))
                                print("model saved to", f'{self.path}/model_{epoch}.pth')
                                torch.save(pi.state_dict(), f'{self.path}/model_{epoch}.pth')
                        mean_rewards[epoch] = tensor_r.mean()
                        v_losses[epoch] = L_v.item()
                        p_losses[epoch] = L_ppo.item()
                        
                # loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
        def sample_trajectories(self):
                sl, al, rl = [], [], []
                for _ in range(20):
                        states = np.zeros((self.max_steps + 1, len(self.train_env.agents), self.train_env.state_len), dtype=float)
                        actions = np.zeros((self.max_steps, len(self.train_env.agents)), dtype=float)
                        rewards = np.zeros((self.max_steps, len(self.train_env.agents)), dtype=float)
                        
                        s = self.train_env.reset()
                        states[0] = s
                        for t in range(self.max_steps):
                                a = self.policy.sample_actions(Variable(torch.tensor(states[t]).float()))
                                s_next, r, _, _ = self.train_env.step(np.array(a))
                                states[t + 1], rewards[t], actions[t] = s_next, r, a  
                        states = torch.tensor(states).float()
                        actions = torch.tensor(actions).float()
                        rewards = torch.tensor(rewards).float()
                        sl.append(states)
                        rl.append(rewards)
                        al.append(actions)
                states = torch.cat(sl, dim=1)
                rewards = torch.cat(rl, dim=1)
                actions = torch.cat(al, dim=1)
                return states, actions, rewards
        
        def ppo_loss(self, advantages, ratio):
                # print("=============== ppo loss ===============")
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                # print(policy_loss_1.max(), policy_loss_2.max())
                # print("========================================")
                return -torch.min(policy_loss_1, policy_loss_2).mean() # mean
                
        def entropy_loss(self, log_prob):
                return -torch.mean(-log_prob)
        
        def compute_gae(self, tensor_r, values, gamma, lambda_):
                delta_t = tensor_r + gamma * values[:, 1:] - values[:, :-1]
                advantages = self.discount_sum(delta_t, gamma * lambda_)
                value_targets = advantages + values[:, :-1]
                return value_targets.squeeze(dim=-1), advantages

        def discount_sum(self, rewards, gamma):
                discounted_sum = 0
                for t in reversed(range(len(rewards))):
                        discounted_sum = rewards[t] + gamma * discounted_sum
                return discounted_sum

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("agent_type")
    parser.add_argument("maps", type=list_of_ints)
    parser.add_argument("policy")
    parser.add_argument("pretrained")
    parser.add_argument("reward", type=int)
    parser.add_argument("steps", type=int)
    parser.add_argument("epochs", type=int)
    parser.add_argument("freq", type=int)

    args = parser.parse_args()

    agent = None
    if "car" in args.agent_type.lower():
        agent = CarLikeBot
        ag_t = "car"
    elif "circle" in args.agent_type.lower():
        agent = CircleRobot
        ag_t = "circle"
    else:
        raise Exception("agent_type must contain car or cicle")

    env = Environment(
            agent, 
            map=args.maps, 
            reward=[MyReward(), MyReward2()][args.reward]
            )

    observation_space = gym.spaces.Box(low=np.zeros(16*4+3), high=np.ones(16*4+3))

    pi = None
    if "less" in args.policy.lower():
        pi = CustomPolicyLessOl(observation_space)
        pi_t = "less"
    elif "std" in args.policy.lower():
        pi = CustomPolicy(observation_space)
        pi_t = "std"
    else:
        raise Exception("policy must contain std or less")
    
    if args.pretrained and  args.pretrained != "None":
        pi.load_state_dict(torch.load(args.pretrained))

    folder_path = f"{ag_t}_{pi_t}_{args.epochs}_{args.reward}"
    os.makedirs(folder_path, exist_ok=True)

    ppo = PPO_gym(
            env,
            pi,
            max_steps=args.steps,
            n_epochs=args.epochs,
            freq = args.freq,
            path = folder_path
            )
    ppo.tarin()
                
                
        
