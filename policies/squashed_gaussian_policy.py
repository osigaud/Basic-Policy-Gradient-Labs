import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
from policies.generic_net import GenericNet

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianPolicy(GenericNet):
    def __init__(self, l1, l2, l3, l4, learning_rate):
        super(SquashedGaussianPolicy, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc_mu = nn.Linear(l3, l4)
        self.fc_std = nn.Linear(l3, l4)
        self.tanh = nn.Tanh()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def log_prob(self, normal_distribution, action):
        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        logp_pi = normal_distribution.log_prob(action).sum(axis=-1)
        val = func.softplus(-2 * action)
        logp_pi -= (2 * (np.log(2) - action - val)).sum(axis=1)
        return logp_pi
    
    def forward(self, state, deterministic=False, with_logprob=True):
        if state.ndim == 1:
            # Add batch dim of 1 before pass in neural network
            state = np.reshape(state, (1, -1))

        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        mu = self.fc_mu(state)
        std = self.fc_std(state)
        log_std = torch.clamp(std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)

        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = self.log_prob(pi_distribution, pi_action)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi


    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            # Forward pass
            pi_action, logp_pi = self.forward(state, deterministic)
            return pi_action

    def train_pg(self, state, action, reward):
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        mu, std = self.forward(state)
        # Negative score function x reward
        # loss = -Normal(mu, std).log_prob(action) * reward
        normal_distribution = Normal(mu, std)
        loss = - self.log_prob(normal_distribution, action) * reward
        self.update(loss)
        return loss

    def train_regress(self, state, action):
        action = torch.FloatTensor(action)
        mu, _ = self.forward(state)
        loss = func.mse_loss(mu, action)
        self.update(loss)
        return loss

    def train_regress_from_batch(self, batch):
        for j in range(batch.size):
            episode = batch.episodes[j]
            state = np.array(episode.state_pool)
            action = np.array(episode.action_pool)
            self.train_regress(state, action)
