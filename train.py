from __future__ import annotations
import numpy as np
from robots import CarLikeBot, CircleRobot
# from random import uniform
from gym_env import Environment
from policies import CustomPolicy 
# from algo.ppo import PPO
from rewards import RewardCircle, MyReward
import gym
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
# torch.autograd.set_detect_anomaly(True)


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
    print("teansor_rrr", tensor_r.shape)
    print("values_sss", values[:, 1:].shape)
    delta_t = tensor_r + gamma * values[:, 1:] - values[:, :-1]
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
        ol[i] = np.array(ol[i])
        ol[i] = torch.tensor(ol[i]).float()
        op[i] = np.array(op[i])
        op[i] = torch.tensor(op[i]).float()

        rewards[i] = np.array(rewards[i])
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
        model_name: str | None = "test_model",
        optimizer = torch.optim.Adam
        ):
        if path:
                pi.load_state_dict(torch.load(path))
        optim = optimizer(pi.parameters(), lr=lr)

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
                for t in ol:
                        if torch.isnan(t).any():
                                print("====\nol contain nan\n====")
                                raise
                for t in op:
                        if torch.isnan(t).any():
                                print("====\nop contain nan\n====")
                                raise
                for t in logp_old:
                        if torch.isnan(t).any():
                                print("====\nlogp_old contain nan\n====")
                                raise
                
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
                                
                        for t in logp:
                                if torch.isnan(t).any():
                                        print("====\nlogp contain nan\n====")
                                        raise

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
                                if torch.isnan(L_v).any():
                                        print("====\nL_v contain nan\n====")
                                        raise
                                p_ratios = torch.exp(logp[i] - logp_old[i])
                                
                                L_ppo = ppo_loss(p_ratios, advantage_estimates[i], epsilon=epsilon)
                                if torch.isnan(L_ppo).any():
                                        print("====\ntL_ppo contain nan\n====")
                                        raise
                                total_loss.append(L_v + L_ppo)
                                
                        for t in total_loss:
                                if torch.isnan(t).any():
                                        print("====\ntotal loss contain nan\n====")
                                        raise
                        # print(total_loss)
                        total_sum_tensor = -torch.mean(torch.stack(total_loss), dim=0)
                        # print(total_sum_tensor)
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
                

def test_sample_trajectories(env, pi, max_steps, i_map):    
    op = [[] for _ in range(len(env.agents))]

    actions = [[] for _ in range(len(env.agents))]  # actions from a(0) to a(T)
    rewards = [[] for _ in range(len(env.agents))]  # rewards from r(0) to r(T)
    env.reset(i_map)
    
    for index, ag in enumerate(env.agents):
        if not ag.collision_w and not ag.collision_a and  not ag.reached:
                _, op_ = ag.history.get_vectors()
                op[index].append(op_)


    indexes = []
    for _ in range(max_steps):

        for index, ag in enumerate(env.agents):
                if not ag.collision_w and not ag.collision_a and  not ag.reached:
                        _, op_ = ag.history.get_vectors()
                        act_ = pi.sample_actions(
                                torch.tensor(np.reshape(op_, (1, 3))).float(),
                                )
                        
                        actions[index].append(act_)

        for index, ag in enumerate(env.agents):
                if not ag.collision_w and not ag.collision_a and not ag.reached:
                        _, op_ = ag.history.get_vectors()
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
        op[i] = np.array(op[i])
        op[i] = torch.tensor(op[i]).float()

        rewards[i] = np.array(rewards[i])
        actions[i] = torch.stack(actions[i])
        rewards[i] = torch.tensor(rewards[i]).float()
    
    return op, actions, rewards

def testPPO(
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
        model_name: str | None = "test_model",
        optimizer = torch.optim.Adam
        ):
        if path:
                pi.load_state_dict(torch.load(path))
        optim = optimizer(pi.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
                op, actions, rewards = test_sample_trajectories(train_env, pi, max_steps, int(np.random.choice(maps)))
                num_ag = len(op)
                with torch.no_grad():
                        logp_old = []
                        for i in range(num_ag):
                                logp_old.append(pi.log_prob(
                                        actions[i][:actions[i].shape[0]],
                                        op[i][:actions[i].shape[0]],
                                        ))
                # for t in ol:
                #         if torch.isnan(t).any():
                #                 print("====\nol contain nan\n====")
                #                 raise
                for t in op:
                        if torch.isnan(t).any():
                                print("====\nop contain nan\n====")
                                raise
                for t in logp_old:
                        if torch.isnan(t).any():
                                print("====\nlogp_old contain nan\n====")
                                raise
                
                for _ in range(sgd_iters):
                        values = []
                        for i in range(num_ag):
                                values.append(pi.value_estimates(
                                        op[i][:actions[i].shape[0]+1]
                                ))

                        logp = []
                        for i in range(num_ag):
                                logp.append(pi.log_prob(
                                        actions[i][:actions[i].shape[0]],
                                        op[i][:actions[i].shape[0]]
                                ))
                                
                        for t in logp:
                                if torch.isnan(t).any():
                                        print("====\nlogp contain nan\n====")
                                        raise

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
                                if torch.isnan(L_v).any():
                                        print("====\nL_v contain nan\n====")
                                        raise
                                p_ratios = torch.exp(logp[i] - logp_old[i])
                                
                                L_ppo = ppo_loss(p_ratios, advantage_estimates[i], epsilon=epsilon)
                                if torch.isnan(L_ppo).any():
                                        print("====\ntL_ppo contain nan\n====")
                                        raise
                                total_loss.append(L_v + L_ppo)
                                
                        for t in total_loss:
                                if torch.isnan(t).any():
                                        print("====\ntotal loss contain nan\n====")
                                        raise
                        # print(total_loss)
                        total_sum_tensor = torch.mean(torch.stack(total_loss), dim=0)
                        # print(total_sum_tensor)
                        optim.zero_grad()
                        total_sum_tensor.backward()
                        optim.step()
                        
                if epoch % test_feq == 0:
                        rewards = 0
                        cnt = 0
                        for _ in range(test_num):
                                _, actions, rew = test_sample_trajectories(train_env, pi, max_steps, int(np.random.choice(maps))) 
                                for i, r in enumerate(rew):
                                        rewards += sum(r)/actions[i].shape[0]
                                rewards /= len(actions)
                                for a in train_env.agents:
                                        if a.reached:
                                                cnt += 1
                        print("epoch:", epoch, "reward:", rewards, "success:", cnt)
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
        
        
class PPO_gym:
        def __init__(
                self,
                train_env,
                policy,
                n_epochs = 100,
                gamma = 0.99,
                lr = 3e-4,
                batch_size = 64,
                max_steps = 126,
                vf_coef: float = 0.5,
                max_grad_norm: float = 0.5,
                gae_lambda: float = 0.95,
                clip_range: float = 0.2,
                ent_coef: float = 0.0
                ):
                self.train_env = train_env
                self.policy = policy
                self.n_epochs = n_epochs
                self.clip_range = clip_range
                self.lr = lr
                self.batch_size = batch_size
                self.gamma = gamma
                self.vf_coef = vf_coef
                self.max_grad_norm = max_grad_norm
                self.gae_lambda = gae_lambda
                self.max_steps = max_steps
                self.ent_coef = ent_coef
                
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
                                
                        if epoch % 20 == 0:
                                print('Epoch %d, mean reward: %.3f, value loss: %.3f, ppo_loss: %.6f' % (epoch, tensor_r.mean(), L_v.item(), L_ppo.item()))
                                torch.save(pi.state_dict(), f'carlike_gym/model_{epoch}.pth')
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
                # print("delta_t", tensor_r.shape, values.shape)
                delta_t = tensor_r + gamma * values[:, 1:] - values[:, :-1]
                # print("delta_t", tensor_r.shape, delta_t.shape, values.shape)
                advantages = self.discount_sum(delta_t, gamma * lambda_)
                # print("compute_gae adv", advantages.shape)
                # print("compute_gae values", values.shape)
                # print("compute_gae values", values[:, :-1].shape)
                value_targets = advantages + values[:, :-1]
                # print("compute_gae value_targets", value_targets.shape)
                # print("compute_gae value_targets -dim", value_targets.squeeze(dim=-1).shape)
                return value_targets.squeeze(dim=-1), advantages

        def discount_sum(self, rewards, gamma):
                # print("=================== disc _ sum ===================")
                
                discounted_sum = 0
                for t in reversed(range(len(rewards))):
                        discounted_sum = rewards[t] + gamma * discounted_sum
                # print(rewards.shape, discounted_sum.shape)
                # print("==================================================")
                return discounted_sum

if __name__ == "__main__":
        env = Environment(CarLikeBot, map=[4,5,6,7], reward=MyReward())
        observation_space = gym.spaces.Box(low=np.zeros(16*4+3), high=np.ones(16*4+3))

        pi = CustomPolicy(observation_space)
        # pi.load_state_dict(torch.load("carlike_gym/model_0.pth"))
        ppo = PPO_gym(env, pi, max_steps=126, n_epochs=1000)
        ppo.tarin()
                
                
        