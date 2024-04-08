import torch 
import numpy as np


def sample_trajectories(env, pi, max_steps):    
    """given an environment env, a stochastic policy pi and number of timesteps T, interact with the environment for T steps 
    using actions sampled from policy. Return torch tensors of collected states, actions and rewards"""
    states = np.zeros((T + 1, N, env.num_states), dtype=float)  # states from s(0) to s(T+1)
    actions = np.zeros((T, N, env.num_actions), dtype=float)  # actions from a(0) to a(T)
    rewards = np.zeros((T, N), dtype=float)  # rewards from r(0) to r(T)
    
    s = env.vector_reset()
    states[0] = s
    for t in range(max_steps):
        a = pi.sample_actions(torch.tensor(states[t]).float())  # policy needs float torch tensor (N, state_dim)
        s_next, r = env.vector_step(np.array(a))  # env needs numpy array of (Nx1)
        states[t + 1], actions[t], rewards[t] = s_next, a, r    
        
    tensor_s = torch.tensor(states).float()   # (T+1, N, state_dim)  care for the extra timestep at the end!
    tensor_a = torch.tensor(actions).float()  # (T, N, 1)
    tensor_r = torch.tensor(rewards).float()  # (T, N)
    
    return tensor_s, tensor_a, tensor_r 
    

def PPO(policy, env):
    max_steps = 128
    epochs = 500
    lr = 0.01
    gamma=0.95
    epsilon = 0.2

    sgd_iters = 5

    # policy, environment and optimizer
    pi = policy()
    train_env = PendulumEnv(config)
    optim = torch.optim.SGD(pi.parameters(), lr=lr)

    mean_rewards, p_losses, v_losses = np.zeros(epochs), np.zeros(epochs), np.zeros(epochs)  # for logging mean rewards over epochs
    for epoch in range(epochs):
        ol, od, og, ov, tensor_a, tensor_r = sample_trajectories(train_env, pi, max_steps)  # collect trajectories using current policy
        # print(tensor_s.shape, tensor_a.shape, tensor_r.shape)
        with torch.no_grad():  # compute the old probabilities
            logp_old = pi.log_prob(tensor_a, tensor_s[:max_steps]).squeeze(2)  # compute log(pi(a_t | s_t))
        
        for i in range(sgd_iters):  # we can even do multiple gradient steps
            values = pi.value_estimates(tensor_s)  # estimate value function for all states 
            logp = pi.log_prob(tensor_a, tensor_s[:max_steps]).squeeze(2)  # compute log(pi(a_t | s_t))

            with torch.no_grad():  # no need for gradients when computing the advantages and value targets
                # value_targets, advantage_estimates = compute_advantage_estimates(tensor_r, values, gamma, bootstrap=True)
                value_targets, advantage_estimates = compute_gae(tensor_r, values, gamma, lambda_=0.97)
                advantage_estimates = (advantage_estimates - advantage_estimates.mean()) / advantage_estimates.std()  # normalize advantages
                
            L_v = value_loss(values[:max_steps], value_targets)  # add the value loss
            
            p_ratios = torch.exp(logp - logp_old)  # compute the ratios r_\theta(a_t | s_t)
            L_ppo = ppo_loss(p_ratios, advantage_estimates, epsilon=epsilon)  # compute the policy gradient loss
            total_loss = L_v + L_ppo
            
            optim.zero_grad()
            total_loss.backward()  # backprop and gradient step
            optim.step()
        
        if epoch % 10 == 0:
            print('Epoch %d, mean reward: %.3f, value loss: %.3f' % (epoch, tensor_r.mean(), L_v.item()))
        mean_rewards[epoch] = tensor_r.mean()
        v_losses[epoch] = L_v.item()
        p_losses[epoch] = L_ppo.item()
        
    train_env.close()

    plot_training(mean_rewards, p_losses, v_losses)
    
    