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

    def forward(self, state, deterministic=False, with_logprob=True):
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
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            print(pi_action)
            val = func.softplus(-2 * pi_action)
            logp_pi -= (2 * (np.log(2) - pi_action - val)).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi


    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            pi_action, logp_pi = self.forward(state, deterministic)
            return pi_action

    def train_pg(self, state, action, reward):
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        mu, std = self.forward(state)
        # Negative score function x reward
        loss = -Normal(mu, std).log_prob(action) * reward
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
