from __future__ import annotations
import numpy as np
from robots import CarLikeBot, CircleRobot
from random import uniform
from env import Environment
from policies import ContiniousPolicy, DiscterePolicy
from algo.ppo import PPO
from rewards import RewardCircle
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)


def sample_trajectories(env, pi, max_steps, i_map):    
    ol = [[] for _ in range(len(env.agents))]
    od = [[] for _ in range(len(env.agents))]
    og = [[] for _ in range(len(env.agents))]
    ov = [[] for _ in range(len(env.agents))]
    actions = [[] for _ in range(len(env.agents))]  # actions from a(0) to a(T)
    rewards = [[] for _ in range(len(env.agents))]  # rewards from r(0) to r(T)
    env.reset(i_map)
    env.get_observation()
    
    for index, ag in enumerate(env.agents):
        if not ag.collision_w and not ag.collision_a and  not ag.reached:
                ol_, od_, og_, ov_ = ag.history.get_vectors()
                ol[index].append(ol_)
                od[index].append(od_)
                og[index].append(og_)
                ov[index].append(ov_)

    indexes = []
    for _ in range(max_steps):

        for index, ag in enumerate(env.agents):
                if not ag.collision_w and not ag.collision_a and  not ag.reached:
                        ol_, od_, og_, ov_ = ag.history.get_vectors()
                        act_ = pi.sample_actions(
                                torch.tensor(np.reshape(ol_, (1, 1, 64))).float(),
                                torch.tensor(np.reshape(od_, (1, 4))).float(),
                                torch.tensor(np.reshape(og_, (1, 8))).float(),
                                torch.tensor(np.reshape(ov_, (1, 8))).float()
                                )
                        
                        actions[index].append(act_)

        for index, ag in enumerate(env.agents):
                if not ag.collision_w and not ag.collision_a and not ag.reached:
                        ol_, od_, og_, ov_ = ag.history.get_vectors()
                        ol[index].append(ol_)
                        od[index].append(od_)
                        og[index].append(og_)
                        ov[index].append(ov_)

        rew, done = env.step(np.array([a[-1] for a in actions]))
        
        for index, ag in enumerate(env.agents):
                if index not in indexes:
                        if ag.collision_w or ag.collision_a or ag.reached:
                                indexes.append(index)
                        rewards[index].append(rew[index])
                        
        if done:
                break
        
    for i in range(len(env.agents)):
        ol[i] = torch.tensor(ol[i]).float()
        od[i] = torch.tensor(od[i]).float()
        og[i] = torch.tensor(og[i]).float()
        ov[i] = torch.tensor(ov[i]).float()

        actions[i] = torch.stack(actions[i])
        rewards[i] = torch.tensor(rewards[i]).float()
    
    return ol, od, og, ov, actions, rewards

def discount_cum_sum(rewards, gamma):
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    for t in range(T):
        # Compute the discounted sum of future rewards from time step t
        returns[t] = torch.sum(rewards[t:] * (gamma ** torch.arange(T - t)))
    return returns

def compute_gae(tensor_r, values, gamma, lambda_):
    delta_t = tensor_r + gamma * values[1:] - values[:-1]
    advantages = discount_cum_sum(delta_t, gamma * lambda_)
    value_targets = advantages + values[:-1]
    return value_targets, advantages

def value_loss(values, value_targets):
    value_loss = torch.tensor(0.)
    T = values.shape[0]
    value_loss += (1/T)*torch.sum((values - value_targets) ** 2)
    return value_loss

def ppo_loss(p_ratios, advantage_estimates, epsilon):
    policy_loss = torch.tensor(0.) 
    T = p_ratios.shape[0]
    
    p_ratios = p_ratios.sum(dim=1)
    clipped_ratios = torch.clamp(p_ratios, 1.0 - epsilon, 1.0 + epsilon)

    surrogate_loss = torch.min(p_ratios * advantage_estimates, clipped_ratios * advantage_estimates)
    policy_loss -= (1/T)*torch.sum(surrogate_loss)
    return policy_loss
    
def plot_training(rewards, p_losses, v_losses=None):
    num_plots = 2 if v_losses is None else 3

    plt.subplot(num_plots, 1, 1)
    plt.plot(rewards, label='mean rewards', color='green')
    plt.ylabel('Mean reward')
    plt.subplot(num_plots, 1, 2)
    plt.plot(p_losses, label='policy loss', color='red')
    plt.ylabel('Policy loss')
    if v_losses is not None:
        plt.subplot(num_plots, 1, 3)
        plt.plot(v_losses, label='value loss', color='blue')
        plt.ylabel('Value loss')
    plt.xlabel('Epoch')
    plt.show()

def compute_advanteges(rewards, values):
        advantages = rewards - values
        return advantages


def PPO(
        train_env,
        pi,
        path: str | None = None,
        max_steps = 512,
        epochs = 5000,
        lr = 0.03,
        gamma=0.95,
        epsilon = 0.2,
        sgd_iters = 10,
        maps = [3],
        test_feq = 100,
        test_num = 25,
        model_name: str | None = "test_model"
        ):
        if path:
                pi.load_state_dict(torch.load(path))
        optim = torch.optim.Adam(pi.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
                ol, od, og, ov, actions, rewards = sample_trajectories(train_env, pi, max_steps, int(np.random.choice(maps)))
                num_ag = len(ol)
                with torch.no_grad():
                        logp_old = []
                        for i in range(num_ag):
                                logp_old.append(pi.log_prob(
                                        actions[i][:actions[i].shape[0]],
                                        ol[i][:actions[i].shape[0]],
                                        od[i][:actions[i].shape[0]],
                                        og[i][:actions[i].shape[0]],
                                        ov[i][:actions[i].shape[0]]
                                        ))

                for _ in range(sgd_iters):
                        values = []
                        for i in range(num_ag):
                                values.append(pi.value_estimates(
                                        ol[i][:actions[i].shape[0]+1],
                                        od[i][:actions[i].shape[0]+1],
                                        og[i][:actions[i].shape[0]+1],
                                        ov[i][:actions[i].shape[0]+1]
                                ))

                        logp = []
                        for i in range(num_ag):
                                logp.append(pi.log_prob(
                                        actions[i][:actions[i].shape[0]],
                                        ol[i][:actions[i].shape[0]],
                                        od[i][:actions[i].shape[0]],
                                        og[i][:actions[i].shape[0]],
                                        ov[i][:actions[i].shape[0]]
                                ))

                        with torch.no_grad():
                                value_targets, advantage_estimates = [], []
                                for i in range(num_ag):
                                        value_t, advantage_e = compute_gae(
                                                rewards[i][:actions[i].shape[0]],
                                                values[i][:actions[i].shape[0]+1],
                                                gamma, lambda_=0.97)
                                        advantage_e = (advantage_e - advantage_e.mean()) / advantage_e.std()
                                        value_targets.append(value_t)
                                        advantage_estimates.append(advantage_e)
                        
                        total_loss = []
                        for i in range(num_ag): 
                                # i = 0
                                # print(f"\n\n{i}\n\n")       
                                L_v = value_loss(values[i][:actions[i].shape[0]], value_targets[i])
                                        
                                p_ratios = torch.exp(logp[i] - logp_old[i])
                                
                                L_ppo = ppo_loss(p_ratios, advantage_estimates[i], epsilon=epsilon)
                                total_loss.append(L_v + L_ppo)
                        
                        total_loss = (total_loss[0] + total_loss[1])/2
                                
                        optim.zero_grad()
                        total_loss.backward()
                        optim.step()
                        
                if epoch % test_feq == 0:
                        rewards = 0
                        for _ in range(test_num):
                                ol, od, og, ov, actions, rew = sample_trajectories(train_env, pi, max_steps, int(np.random.choice(maps))) 
                                for i, r in enumerate(rew):
                                        rewards += sum(r)/actions[i].shape[0]
                                rewards /= len(actions)
                        print("epoch:", epoch, "reward:", rewards)
                        if model_name:
                                torch.save(pi.state_dict(), f'{model_name}_{epoch}.pth')
        
        if model_name:
                torch.save(pi.state_dict(), f'{model_name}_final.pth')
                

def my_sample_trajectories(env, pi, max_steps, i_map):    
    ol = [[] for _ in range(len(env.agents))]
    op = [[] for _ in range(len(env.agents))]

    actions = [[] for _ in range(len(env.agents))]  # actions from a(0) to a(T)
    rewards = [[] for _ in range(len(env.agents))]  # rewards from r(0) to r(T)
    env.reset(i_map)
    
    for index, ag in enumerate(env.agents):
        if not ag.collision_w and not ag.collision_a and  not ag.reached:
                ol_, op_ = ag.history.get_vectors()
                ol[index].append(ol_)
                op[index].append(op_)


    indexes = []
    for _ in range(max_steps):

        for index, ag in enumerate(env.agents):
                if not ag.collision_w and not ag.collision_a and  not ag.reached:
                        ol_, op_ = ag.history.get_vectors()
                        act_ = pi.sample_actions(
                                torch.tensor(np.reshape(ol_, (1, 4, 16))).float(),
                                torch.tensor(np.reshape(op_, (1, 3))).float(),
                                )
                        
                        actions[index].append(act_)

        for index, ag in enumerate(env.agents):
                if not ag.collision_w and not ag.collision_a and not ag.reached:
                        ol_, op_ = ag.history.get_vectors()
                        ol[index].append(ol_)
                        op[index].append(op_)
                        
        rew, done = env.step(np.array([a[-1] for a in actions]))
        
        for index, ag in enumerate(env.agents):
                if index not in indexes:
                        if ag.collision_w or ag.collision_a or ag.reached:
                                indexes.append(index)
                        rewards[index].append(rew[index])
                        
        if done:
                break
        
    for i in range(len(env.agents)):
        ol[i] = torch.tensor(ol[i]).float()
        op[i] = torch.tensor(op[i]).float()

        actions[i] = torch.stack(actions[i])
        rewards[i] = torch.tensor(rewards[i]).float()
    
    return ol, op, actions, rewards

def MyPPO(
        train_env,
        pi,
        path: str | None = None,
        max_steps = 512,
        epochs = 5000,
        lr = 0.03,
        gamma=0.95,
        epsilon = 0.2,
        sgd_iters = 10,
        maps = [3],
        test_feq = 100,
        test_num = 25,
        model_name: str | None = "test_model"
        ):
        if path:
                pi.load_state_dict(torch.load(path))
        optim = torch.optim.Adam(pi.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
                ol, op, actions, rewards = my_sample_trajectories(train_env, pi, max_steps, int(np.random.choice(maps)))
                num_ag = len(ol)
                with torch.no_grad():
                        logp_old = []
                        for i in range(num_ag):
                                logp_old.append(pi.log_prob(
                                        actions[i][:actions[i].shape[0]],
                                        ol[i][:actions[i].shape[0]],
                                        op[i][:actions[i].shape[0]],
                                        ))

                for _ in range(sgd_iters):
                        values = []
                        for i in range(num_ag):
                                values.append(pi.value_estimates(
                                        ol[i][:actions[i].shape[0]+1],
                                        op[i][:actions[i].shape[0]+1]
                                ))

                        logp = []
                        for i in range(num_ag):
                                logp.append(pi.log_prob(
                                        actions[i][:actions[i].shape[0]],
                                        ol[i][:actions[i].shape[0]],
                                        op[i][:actions[i].shape[0]]
                                ))

                        with torch.no_grad():
                                value_targets, advantage_estimates = [], []
                                for i in range(num_ag):
                                        value_t, advantage_e = compute_gae(
                                                rewards[i][:actions[i].shape[0]],
                                                values[i][:actions[i].shape[0]+1],
                                                gamma, lambda_=0.97)
                                        advantage_e = (advantage_e - advantage_e.mean()) / advantage_e.std()
                                        value_targets.append(value_t)
                                        advantage_estimates.append(advantage_e)
                        
                        total_loss = []
                        for i in range(num_ag):      
                                L_v = value_loss(values[i][:actions[i].shape[0]], value_targets[i])
                                        
                                p_ratios = torch.exp(logp[i] - logp_old[i])
                                
                                L_ppo = ppo_loss(p_ratios, advantage_estimates[i], epsilon=epsilon)
                                total_loss.append(L_v + L_ppo)
                        
                        total_sum_tensor = -torch.sum(torch.stack(total_loss), dim=0)/len(total_loss)
                        optim.zero_grad()
                        total_sum_tensor.backward()
                        optim.step()
                        
                if epoch % test_feq == 0:
                        rewards = 0
                        for _ in range(test_num):
                                _, _, actions, rew = my_sample_trajectories(train_env, pi, max_steps, int(np.random.choice(maps))) 
                                for i, r in enumerate(rew):
                                        rewards += sum(r)/actions[i].shape[0]
                                rewards /= len(actions)
                        print("epoch:", epoch, "reward:", rewards)
                        if model_name:
                                torch.save(pi.state_dict(), f'{model_name}_{epoch}.pth')
        
        if model_name:
                torch.save(pi.state_dict(), f'{model_name}_final.pth')
                
                
# def A2C(
#         train_env,
#         pi,
#         path: str | None = None,
#         max_steps = 512,
#         epochs = 5000,
#         lr = 0.03,
#         gamma=0.95,
#         epsilon = 0.2,
#         sgd_iters = 10,
#         maps = [3],
#         test_feq = 100,
#         test_num = 25,
#         model_name: str | None = "test_model"
#         ):
        
        
