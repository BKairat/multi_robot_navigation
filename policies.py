import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from torch.distributions import Categorical
from torchsummary import summary

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomCNN, self).__init__(observation_space, features_dim=320)
        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=4, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=1)
        # Define fully connected layers for other inputs
        self.fc_op = nn.Linear(3, 32)
        # Merge layers
        self.fc_merge = nn.Linear(352, 352)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract laser scanner observations
        ol = observations[:, :16*4]  # Assuming the first 4 observations are for laser scanner
        # Process laser scanner observations through convolutional layers
        ol = torch.reshape(ol, (ol.shape[0], 4, 16)) 
        ol = F.relu(self.conv1(ol))
        ol = F.relu(self.conv2(ol))
        ol = ol.view(ol.size(0), -1)  # Flatten the output
        
        # Process other observations through fully connected layers
        op = observations[:, 16*4:]  # Assuming the 5th observation is for od
        op = F.relu(self.fc_op(op))
        # Merge all features
        merged = torch.cat((ol, op), dim=1)
        
        # Process merged features through final fully connected layer
        merged = F.relu(self.fc_merge(merged))
        return merged

class CustomCNNLessOl(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomCNNLessOl, self).__init__(observation_space, features_dim=64)
        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, stride=2)
        # Define fully connected layers for other inputs
        self.fc_op = nn.Linear(3, 32)
        # Merge layers
        self.fc_merge = nn.Linear(64, 64)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract laser scanner observations
        ol = observations[:, :16*4]  # Assuming the first 4 observations are for laser scanner
        # Process laser scanner observations through convolutional layers
        ol = torch.reshape(ol, (ol.shape[0], 4, 16))
        ol = F.relu(self.conv1(ol))
        ol = F.relu(self.conv2(ol))
        ol = ol.view(ol.size(0), -1)  # Flatten the output
        # Process other observations through fully connected layers
        op = observations[:, 16*4:]  # Assuming the 5th observation is for od
        op = F.relu(self.fc_op(op))
        # print(f"ol: {ol.shape}\nop: {op.shape}")
        # Merge all features
        merged = torch.cat((ol, op), dim=1)

        # Process merged features through final fully connected layer
        merged = F.relu(self.fc_merge(merged))
        return merged

# Define custom policy
class CustomPolicy(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomPolicy, self).__init__()
        # Define custom feature extractor
        self.features_extractor = CustomCNN(observation_space)
        # Define actor and critic networks
        self.actor = nn.Linear(352, 10)
        self.critic = nn.Linear(352, 1)
    
    def forward(self, observations: torch.Tensor):
        # Extract features using custom feature extractor
        features = self.features_extractor(observations)
        # Compute action probabilities
        action_probs = F.softmax(self.actor(features), dim=-1)
        # Compute state value
        state_value = self.critic(features)
        return action_probs, state_value
    
    def value_estimates(self, observations):
        val = []
        for i in range(observations.shape[1]):
            features = self.features_extractor(observations[:, i, :])
            val.append(self.critic(features))
        values = torch.stack(val, dim = 0)
        return values
    
    def log_prob(self, actions, states):
        action_probs = []
        for i in range(states.shape[1]):
            features = self.features_extractor(states[:, i, :])
            action_probs.append(F.softmax(self.actor(features), dim=-1))
        action_probs = torch.stack(action_probs, dim = 0)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions.transpose(0,1))
        return log_probs 
    
    def sample_actions(self, observations: torch.Tensor):
        features = self.features_extractor(observations)
        action_probs = F.softmax(self.actor(features), dim=-1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample()
        return actions

class CustomPolicyLessOl(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomPolicyLessOl, self).__init__()
        # Define custom feature extractor
        self.features_extractor = CustomCNNLessOl(observation_space)
        # Define actor and critic networks
        self.actor = nn.Sequential( 
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10),
                )
        self.critic = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
                )

    def forward(self, observations: torch.Tensor):
        # Extract features using custom feature extractor
        features = self.features_extractor(observations)
        # Compute action probabilities
        action_probs = F.softmax(self.actor(features), dim=-1)
        # Compute state value
        state_value = self.critic(features)
        return action_probs, state_value

    def value_estimates(self, observations):
        val = []
        for i in range(observations.shape[1]):
            features = self.features_extractor(observations[:, i, :])
            val.append(self.critic(features))
        values = torch.stack(val, dim = 0)
        return values

    def log_prob(self, actions, states):
        action_probs = []
        for i in range(states.shape[1]):
            features = self.features_extractor(states[:, i, :])
            action_probs.append(F.softmax(self.actor(features), dim=-1))
        action_probs = torch.stack(action_probs, dim = 0)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions.transpose(0,1))
        return log_probs

    def sample_actions(self, observations: torch.Tensor):
        features = self.features_extractor(observations)
        action_probs = F.softmax(self.actor(features), dim=-1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample()
        return actions

if __name__ == "__main__":
    observation_space = gym.spaces.Box(low=np.zeros(16*4+3), high=np.ones(16*4+3))

    # Sample an observation from the space
    observation = np.array([observation_space.sample() for _ in range(10)])

    # Create custom policy for PPO algorithm
    policy = CustomPolicyLessOl(observation_space)
    observation = torch.tensor(observation)
    policy.sample_actions(observation)
