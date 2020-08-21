import torch
import torch.nn as nn
import numpy as np
from critics.critic_network import CriticNetwork


class VNetwork(CriticNetwork):
    """
    A value function critic network
    """
    def __init__(self, l1, l2, l3, l4, learning_rate):
        super(VNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        """
        Compute the value from a state, going through the network
        :param state: the given state(s)
        :return: the corresponding values, as a torch tensor
        """
        state = torch.from_numpy(state).float()
        value = self.relu(self.fc1(state))
        value = self.relu(self.fc2(value))
        value = self.fc3(value)
        return value


    def evaluate(self, state, action=None):
        """
         Return the critic value at a given state, as a numpy structure
         :param state: the given state
         :param action: a given action. Should not be specified, added as a parameter to be consistent with Q-networks
         :return: the value
         """
        x = self.forward(state)
        return x.data.numpy()

    def compute_bootstrap_target(self, reward, done, next_state, next_action, gamma):
        """
        Compute the target value using the bootstrap (Bellman backup) equation
        The target is then used to train the critic
        :param reward: the reward value in the sample(s)
        :param done: whether this is the final step
        :param next_state: the next state in the sample(s)
        :param next_action: the next action. Should not be specified, added as a parameter to be consistent with Q-networks
        :param gamma: the discount factor
        :return: the target value
        """
        next_value = np.concatenate(self.forward(next_state).data.numpy())
        delta = reward + gamma * (1 - done) * next_value
        return delta

    def compute_loss_to_target(self, state, action, target):
        """
        Compute the MSE between a target value and the critic value for the state action pair(s)
        :param state: a state or vector of state
        :param action: an action. Should not be specified, added as a parameter to be consistent with Q-networks
        :param target: the target value
        :return: the resulting loss
        """
        val = self.forward(state)
        return self.loss_func(val, target)
