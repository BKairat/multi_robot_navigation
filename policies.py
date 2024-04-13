import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

        
class ContiniousPolicy(nn.Module):
    def __init__(self):
        super(ContiniousPolicy, self).__init__()

        # Convolutional layers for ol (laser scanner input)
        self.conv_layers_ol = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.Tanh()
        )

        # Fully connected layers for od, og, ov
        self.fc_od = nn.Sequential(
            nn.Linear(in_features=4, out_features=16),
            nn.Tanh()
        )
        self.fc_og = nn.Sequential(
            nn.Linear(in_features=8, out_features=32),
            nn.Tanh()
        )
        self.fc_ov = nn.Sequential(
            nn.Linear(in_features=8, out_features=32),
            nn.Tanh()
        )

        # Fully connected layer to merge all inputs
        self.fc_merge = nn.Sequential(
            nn.Linear(in_features=1936, out_features=264),
            nn.Tanh()
        )

        # Final output layer for velocity
        self.fc_velocity = nn.Linear(in_features=264, out_features=4)
        self.fc_value = nn.Linear(in_features=264, out_features=1)

    def forward(self, ol, od, og, ov):
        # Forward pass for ol
        ol = self.conv_layers_ol(ol)
        ol = torch.flatten(ol, start_dim=1)

        # Forward pass for od, og, ov
        od = self.fc_od(od)
        og = self.fc_og(og)
        ov = self.fc_ov(ov)

        # Concatenate all the fully connected layers
        merged = torch.cat((ol, od, og, ov), dim=1)

        # Apply the final fully connected layer
        merged = self.fc_merge(merged)

        # Output layer for velocity
        velocity = self.fc_velocity(merged)
        # value = self.fc_value(merged)

        return velocity
    
    def value(self, ol, od, og, ov):
        # Forward pass for ol
        ol = self.conv_layers_ol(ol)
        ol = torch.flatten(ol, start_dim=1)

        # Forward pass for od, og, ov
        od = self.fc_od(od)
        og = self.fc_og(og)
        ov = self.fc_ov(ov)

        # Concatenate all the fully connected layers
        merged = torch.cat((ol, od, og, ov), dim=1)

        # Apply the final fully connected layer
        merged = self.fc_merge(merged)

        # Output layer for velocity
        value = self.fc_value(merged)
        return value
    
    def determine_actions(self, ol, od, og, ov):
        params = self.forward(ol, od, og, ov)  # map states to distribution parameters
        mu, _ = torch.chunk(params, 2, -1)  # split the parameters into mean and std, return mean
        return mu
        
    def sample_actions(self, ol, od, og, ov):
        params = self.forward(ol, od, og, ov)
        mu, sigma = torch.chunk(params, 2, -1)
        sigma = torch.nn.functional.softplus(sigma)
        distribution = Normal(mu, sigma)  # create distribution of size (T, N, action_dim)
        actions = distribution.sample()  # sample actions
        if tuple(actions.shape) == (1, 2):
            actions = actions.view(2)
        return actions
    
    def log_prob(self, actions, ol, od, og, ov):
        params = self.forward(ol, od, og, ov)  # map states to distribution parameters
        mu, sigma = torch.chunk(params, 2, -1)  # split the parameters into mean and std
        sigma = torch.nn.functional.softplus(sigma)  # make sure std is positive
        distribution = Normal(mu, sigma)  # create distribution of size (T, N, action_dim)
        logp = distribution.log_prob(actions)
        if len(logp.shape) == 3 and logp.shape[2] > 1:  # this allows generalization to multi-dim action spaces
            logp = logp.sum(dim=2, keepdim=True)  # sum over the action dimension
        return logp
    
    def value_estimates(self, ol, od, og, ov):
        return self.value(ol, od, og, ov).squeeze()
    
    
class DiscterePolicy(nn.Module):
    def __init__(self):
        super(DiscterePolicy, self).__init__()

        # Convolutional layers for ol (laser scanner input)
        self.conv_layers_ol = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Fully connected layers for od, og, ov
        self.fc_od = nn.Sequential(
            nn.Linear(in_features=4, out_features=16),
            nn.ReLU()
        )
        self.fc_og = nn.Sequential(
            nn.Linear(in_features=8, out_features=32),
            nn.ReLU()
        )
        self.fc_ov = nn.Sequential(
            nn.Linear(in_features=8, out_features=32),
            nn.ReLU()
        )

        # Fully connected layer to merge all inputs
        self.fc_merge = nn.Sequential(
            nn.Linear(in_features=1936, out_features=264),
            nn.Tanh()
        )

        # Final output layer for velocity
        self.fc_velocity = nn.Sequential(
            nn.Linear(in_features=264, out_features=10),
            nn.Softmax(dim=1)
        )
        self.fc_value = nn.Linear(in_features=264, out_features=1)

    def forward(self, ol, od, og, ov):
        # Forward pass for ol
        ol = self.conv_layers_ol(ol)
        ol = torch.flatten(ol, start_dim=1)

        # Forward pass for od, og, ov
        od = self.fc_od(od)
        og = self.fc_og(og)
        ov = self.fc_ov(ov)

        # Concatenate all the fully connected layers
        merged = torch.cat((ol, od, og, ov), dim=1)

        # Apply the final fully connected layer
        merged = self.fc_merge(merged)

        # Output layer for velocity
        velocity = self.fc_velocity(merged)
        # value = self.fc_value(merged)

        return velocity
    
    def value(self, ol, od, og, ov):
        # Forward pass for ol
        ol = self.conv_layers_ol(ol)
        ol = torch.flatten(ol, start_dim=1)

        # Forward pass for od, og, ov
        od = self.fc_od(od)
        og = self.fc_og(og)
        ov = self.fc_ov(ov)

        # Concatenate all the fully connected layers
        merged = torch.cat((ol, od, og, ov), dim=1)

        # Apply the final fully connected layer
        merged = self.fc_merge(merged)

        # Output layer for velocity
        value = self.fc_value(merged)
        return value
    
    def determine_actions(self, ol, od, og, ov):
        params = self.forward(ol, od, og, ov)  # map states to distribution parameters
        mu, _ = torch.chunk(params, 2, -1)  # split the parameters into mean and std, return mean
        return mu
        
    def sample_actions(self, ol, od, og, ov):
        params = self.forward(ol, od, og, ov)
        max_values = torch.argmax(params, dim=1)
        max_values = max_values.squeeze()
        return max_values
    
    def log_prob(self, actions, ol, od, og, ov):
        params = self.forward(ol, od, og, ov)
        logp = torch.log(params)
        return logp
    
    def value_estimates(self, ol, od, og, ov):
        return self.value(ol, od, og, ov).squeeze()
    
    
class ContiniousPolicy001(nn.Module):
    def __init__(self):
        super(ContiniousPolicy001, self).__init__()

        # Convolutional layers for ol (laser scanner input)
        self.conv_layers_ol = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            nn.Tanh()
        )

        # Fully connected layers for od, og, ov
        self.fc_op = nn.Sequential(
            nn.Linear(in_features=3, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=16),
            nn.Tanh()
        )

        # Fully connected layer to merge all inputs
        self.fc_merge = nn.Sequential(
            nn.Linear(in_features=208, out_features=208),
            nn.Tanh()
        )

        # Final output layer for velocity
        self.fc_velocity = nn.Sequential(
            nn.Linear(in_features=208, out_features=126),
            nn.Tanh(),
            nn.Linear(in_features=126, out_features=4),
            nn.Tanh()
        )
        self.fc_value = nn.Sequential(
            nn.Linear(in_features=208, out_features=126),
            nn.Tanh(),
            nn.Linear(in_features=126, out_features=1),
            nn.Tanh()
        )

    def forward(self, ol, op):
        # Forward pass for ol
        ol = self.conv_layers_ol(ol)
        ol = torch.flatten(ol, start_dim=1)

        # Forward pass for od, og, ov
        op = self.fc_op(op)

        # Concatenate all the fully connected layers
        merged = torch.cat((ol, op), dim=1)

        # Apply the final fully connected layer
        merged = self.fc_merge(merged)

        # Output layer for velocity
        velocity = self.fc_velocity(merged)
        # value = self.fc_value(merged)

        return velocity
    
    def value(self, ol, op):
        # Forward pass for ol
        ol = self.conv_layers_ol(ol)
        ol = torch.flatten(ol, start_dim=1)

        # Forward pass for od, og, ov
        op = self.fc_op(op)

        # Concatenate all the fully connected layers
        merged = torch.cat((ol, op), dim=1)

        # Apply the final fully connected layer
        merged = self.fc_merge(merged)

        # Output layer for velocity
        value = self.fc_value(merged)
        return value

    def determine_actions(self, ol, op):
        params = self.forward(ol, op)  # map states to distribution parameters
        mu, _ = torch.chunk(params, 2, -1)  # split the parameters into mean and std, return mean
        return mu
        
    def sample_actions(self, ol, op):
        params = self.forward(ol, op)
        mu, sigma = torch.chunk(params, 2, -1)
        sigma = torch.nn.functional.softplus(sigma)
        distribution = Normal(mu, sigma)  # create distribution of size (T, N, action_dim)
        actions = distribution.sample()  # sample actions
        if tuple(actions.shape) == (1, 2):
            actions = actions.view(2)
        return actions
    
    def log_prob(self, actions, ol, op):
        params = self.forward(ol, op)  # map states to distribution parameters
        mu, sigma = torch.chunk(params, 2, -1)  # split the parameters into mean and std
        sigma = torch.nn.functional.softplus(sigma)  # make sure std is positive
        distribution = Normal(mu, sigma)  # create distribution of size (T, N, action_dim)
        logp = distribution.log_prob(actions)
        if len(logp.shape) == 3 and logp.shape[2] > 1:  # this allows generalization to multi-dim action spaces
            logp = logp.sum(dim=2, keepdim=True)  # sum over the action dimension
        return logp
    
    def value_estimates(self, ol, op):
        return self.value(ol, op).squeeze()