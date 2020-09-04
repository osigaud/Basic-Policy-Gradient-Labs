import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
from policies.generic_net import GenericNet


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def log_prob(normal_distribution, action):
    """
    Compute the log probability of an action from a Gaussian distribution
    This function performs the necessary corrections in the computation
    to take into account the presence of tanh in the squashed Gaussian function
    see https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
    for details
    :param normal_distribution: the Gaussian distribution used to draw an action
    :param action: the action whose probability must be estimated
    :return: the obtained log probability
    """
    logp_pi = normal_distribution.log_prob(action).sum(axis=-1)
    val = func.softplus(-2 * action)
    logp_pi -= (2 * (np.log(2) - action - val)).sum(axis=1)
    return logp_pi


class SquashedGaussianPolicy(GenericNet):
    """
      A policy whose probabilistic output is drawn from a squashed Gaussian function
      """
    def __init__(self, l1, l2, l3, l4, learning_rate):
        super(SquashedGaussianPolicy, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc_mu = nn.Linear(l3, l4)
        self.fc_std = nn.Linear(l3, l4)
        self.tanh_layer = nn.Tanh()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, state):
        """
        Compute the pytorch tensors resulting from sending a state or vector of states through the policy network
        The obtained tensors can be used to obtain an action by calling select_action
        :param state: the input state(s)
        :return: the resulting pytorch tensor (here the max and standard deviation of a Gaussian probability of action)
        """
        # To deal with numpy's poor behavior for one-dimensional vectors
        # Add batch dim of 1 before sending through the network
        if state.ndim == 1:
            state = np.reshape(state, (1, -1))
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        mu = self.fc_mu(state)
        std = self.fc_std(state)
        log_std = torch.clamp(std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def select_action(self, state, deterministic=False):
        """
        Compute an action or vector of actions given a state or vector of states
        :param state: the input state(s)
        :param deterministic: whether the policy should be considered deterministic or not
        :return: the resulting action(s)
        """
        with torch.no_grad():
            # Forward pass
            mu, std = self.forward(state)
            pi_distribution = Normal(mu, std)

            if deterministic:
                # Only used for evaluating policy at test time.
                pi_action = mu
            else:
                pi_action = pi_distribution.rsample()

            # Finally applies tanh for squashing
            pi_action = torch.tanh(pi_action)
            if len(pi_action) == 1:
                pi_action = pi_action[0]
            return pi_action.data.numpy().astype(float)

    def train_pg(self, state, action, reward):
        """
        Train the policy using a policy gradient approach
        :param state: the input state(s)
        :param action: the input action(s)
        :param reward: the resulting reward
        :return: the loss applied to train the policy
        """
        act = torch.FloatTensor(action)
        rwd = torch.FloatTensor(reward)
        mu, std = self.forward(state)
        # Negative score function x reward
        # loss = -Normal(mu, std).log_prob(action) * reward
        normal_distribution = Normal(mu, std)
        loss = - log_prob(normal_distribution, act) * rwd
        self.update(loss)
        return loss

    def train_regress(self, state, action, estimation_method='mse'):
        """
        Train the policy to perform the same action(s) in the same state(s) using regression
        :param state: the input state(s)
        :param action: the input action(s)
        :param estimation_method: whther we use mse or log_likelihood
        :return: the loss applied to train the policy
        """
        assert estimation_method in ['mse', 'log_likelihood'], 'unsupported estimation method'
        action = torch.FloatTensor(action)
        mu, std = self.forward(state)
        if estimation_method == 'mse':
            loss = func.mse_loss(mu, action)
        else:
            normal_distribution = Normal(mu, std)
            loss = -log_prob(normal_distribution, action.view(-1, 1))
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
