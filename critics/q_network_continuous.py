import torch
import torch.nn as nn
import numpy as np
from critics.critic_network import CriticNetwork


class QNetworkContinuous(CriticNetwork):
    def __init__(self, l1, l2, l3, l4, learning_rate):
        super(QNetworkContinuous, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state, action):
        """
         Compute the value from a state action pair, going through the network
         :param action: the chosen action
         :param state: the given state(s)
         :return: the corresponding values, as a torch tensor
         """
        # added by Thomas
        action = np.reshape(action, (-1, 1))
        if state.ndim == 1:
            # Add batch dim of 1 before pass in neural network
            state = np.reshape(state, (1, -1))
        # end of added by Thomas
        x = np.hstack((state, action))
        state = torch.from_numpy(x).float()
        value = self.relu(self.fc1(state))
        value = self.relu(self.fc2(value))
        value = self.fc3(value)
        return value

    def evaluate(self, state, action):
        """
        Return the critic value at a state action pair, as a numpy structure
        :param state: the given state
        :param action: the given action
        :return: the value
        """
        x = self.forward(state, action)
        return x.data.numpy()

    def compute_bootstrap_target(self, reward, done, next_state, next_action, gamma):
        """
        Compute the target value using the bootstrap (Bellman backup) equation
        The target is then used to train the critic
        :param reward: the reward value in the sample(s)
        :param done: whether this is the final step
        :param next_state: the next state in the sample(s)
        :param next_action: the next action in the sample(s) (used for SARSA)
        :param gamma: the discount factor
        :return: the target value
        """
        next_value = np.concatenate(self.forward(next_state, next_action).data.numpy())
        return reward + gamma * (1 - done) * next_value

    def compute_loss_to_target(self, state, action, target):
        """
        Compute the MSE between a target value and the critic value for the state action pair(s)
        :param state: a state or vector of state
        :param action: an action or vector of actions
        :param target: the target value
        :return: the resulting loss
        """
        val = self.forward(state, action)
        return self.loss_func(val, target)
