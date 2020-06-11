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
        x = np.hstack((state, action))
        state = torch.from_numpy(x).float()
        value = self.relu(self.fc1(state))
        value = self.relu(self.fc2(value))
        value = self.fc3(value)
        return value

    def evaluate(self, state, action):
        x = self.forward(state, action)
        y = x.data.numpy()
        return y

    def compute_bootstrap_target(self, reward, done, next_state, next_action, gamma):
        nextvalue = np.concatenate(self.forward(next_state, next_action).data.numpy())
        return reward + gamma * (1 - done) * nextvalue

    def compute_target_loss(self, state, action, target, train):
        val = self.forward(state, action)
        value_loss = self.loss_func(val, target)
        if train:
            self.update(value_loss)
        return value_loss
