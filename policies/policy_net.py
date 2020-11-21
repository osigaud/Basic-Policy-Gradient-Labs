import numpy as np
import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from policies import GenericNet


class PolicyNet(GenericNet):
    def __init__(self, learning_rate, init_alpha=0.01, lr_alpha=0.001, target_entropy_alpha=-1.0):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.target_entropy_alpha = target_entropy_alpha
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        self.losses = None

    def forward(self, state):
        state = F.relu(self.fc1(state))
        mu = self.fc_mu(state)
        std = F.softplus(self.fc_std(state))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, a, r, s_prime, done = mini_batch
        a_prime, log_prob = self.forward(s_prime)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a_prime), q2(s, a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy  # for gradient ascent
        self.losses = loss.data.numpy().astype(float).mean()
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy_alpha).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def select_action(self, state, deterministic=False):
        """
        Compute an action or vector of actions given a state or vector of states
        :param state: the input state(s)
        :param deterministic: whether the policy should be considered deterministic or not
        :return: the resulting action(s)
        """
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            x = F.relu(self.fc1(state))
            mu = self.fc_mu(x)
            std = F.softplus(self.fc_std(x))
            if deterministic:
                pi_action = mu
            else:
                dist = Normal(mu, std)
                action = dist.rsample()
                pi_action = action
            pi_action = torch.tanh(pi_action)
            return pi_action.data.numpy().astype(float)
