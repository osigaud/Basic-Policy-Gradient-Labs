import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
from policies.generic_net import GenericNet


class NormalPolicy(GenericNet):
    """
     A policy whose probabilistic output is drawn from a Gaussian function
     """
    def __init__(self, l1, l2, l3, l4, learning_rate):
        super(NormalPolicy, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc_mu = nn.Linear(l3, l4)
        self.fc_std = nn.Linear(l3, l4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        """
         Compute the pytorch tensors resulting from sending a state or vector of states through the policy network
         The obtained tensors can be used to obtain an action by calling select_action
         :param state: the input state(s)
         :return: the resulting pytorch tensor (here the max and standard deviation of a Gaussian probability of action)
         """
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        mu = self.fc_mu(state)
        std = 1.5  # 20*self.softplus(self.fc_std(state))
        return mu, std

    def select_action(self, state, deterministic=False):
        """
        Compute an action or vector of actions given a state or vector of states
        :param state: the input state(s)
        :param deterministic: whether the policy should be considered deterministic or not
        :return: the resulting action(s)
        """
        with torch.no_grad():
            mu, std = self.forward(state)
            if deterministic:
                return mu.data.numpy().astype(float)
            else:
                n = Normal(mu, std)
                action = n.sample()
            return action.data.numpy().astype(float)

    def train_pg(self, state, action, reward):
        """
        Train the policy using a policy gradient approach
        :param state: the input state(s)
        :param action: the input action(s)
        :param reward: the resulting reward
        :return: the loss applied to train the policy
        """
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward) 
        mu, std = self.forward(state)
        # Negative score function x reward
        loss = -Normal(mu, std).log_prob(action) * reward  
        self.update(loss)
        return loss

    def train_regress(self, state, action, estimation_method='log_likelihood'):
        """
         Train the policy to perform the same action(s) in the same state(s) using regression
         :param state: the input state(s)
         :param action: the input action(s)
         :return: the loss applied to train the policy
         """
        assert estimation_method in ['mse', 'log_likelihood'], 'unsupported estimation method'
        action = torch.FloatTensor(action)
        mu, std = self.forward(state)
        if estimation_method == 'mse':
            loss = func.mse_loss(mu, action)
        else:
            normal_distribution = Normal(mu, std)
            loss = -normal_distribution.log_prob(action)
        self.update(loss)
        return loss

    def train_regress_from_batch(self, batch) -> None:
        """
        Train the policy using a policy gradient approach from a full batch of episodes
        :param batch: the batch used for training
        :return: nothing
        """
        for j in range(batch.size):
            episode = batch.episodes[j]
            state = np.array(episode.state_pool)
            action = np.array(episode.action_pool)
            self.train_regress(state, action)
