import numpy as np
import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from policies import GenericNet


class PolicyNet(GenericNet):
    def __init__(self, learning_rate, init_alpha=0.01, lr_alpha=0.001, target_entropy_alpha=-1.0, state_size=3,
                 hidden_layer=128):
        """Build a squashed gaussian politic learn the mean and the standard deviation of the normal distribution.

        Args:
            learning_rate (float): The learning rate.
            init_alpha (float): The initial value of alpha.
            lr_alpha (float): The learning rate of alpha.
            target_entropy_alpha (float): The target entropy of alpha.
            state_size (int): The number of elements in a single state vector.
            hidden_layer (int): The number of neurons of the hidden layer (should be even).

        Attributes:
            losses (torch.Tensor): The latest computed losses otherwise None.
        """
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer)
        self.fc_mu = nn.Linear(hidden_layer, 1)
        self.fc_std = nn.Linear(hidden_layer, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.target_entropy_alpha = target_entropy_alpha
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        self.losses = None

    def rescale_action(self, action):
        return 2.0 * action

    def forward(self, state):
        """Forward the state through the network.

        Args:
           state (torch.Tensor): The column tensor of state(s).

        Returns:
           torch.Tensor: The action to perform.
        """
        state = F.relu(self.fc1(state))
        mu = self.fc_mu(state)
        std = F.softplus(self.fc_std(state))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = self.rescale_action(torch.tanh(action))  # Rescale the action space to match the environment
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def select_action(self, state, deterministic=False):
        """Compute an action or vector of actions given a state or vector of states.

        Args:
            state (numpy.ndarray): The input state(s).
            deterministic (bool): Whether the policy should be considered deterministic or not.

        Returns:
            numpy.ndarray: The resulting action(s).
        """
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            x = F.relu(self.fc1(state))
            mu = self.fc_mu(x)
            std = F.softplus(self.fc_std(x))
            if deterministic:
                action = mu
            else:
                dist = Normal(mu, std)
                action = dist.rsample()
            real_action = self.rescale_action(torch.tanh(action))  # Rescale the action space to match the environment
            return real_action.data.numpy().astype(float)

    def train_net(self, critic, mini_batch):
        """Train the network.

        Args:
            critic (critics.DoubleQNet): The critic.
            mini_batch (torch.Tensor): The mini batch of data containing the state, the action, the reward,
            the next state and 1 - done in each row.
        """
        state, action, _, next_state, _ = mini_batch
        action_policy, log_prob = self.forward(next_state)
        entropy = -self.log_alpha.exp() * log_prob

        q_value = critic.forward(state, action_policy)

        loss = -q_value - entropy  # for gradient ascent
        self.losses = loss.data.numpy().astype(float).mean()
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        loss_alpha = -(self.log_alpha.exp() * (log_prob + self.target_entropy_alpha).detach()).mean()
        loss_alpha.backward()
        self.log_alpha_optimizer.step()
