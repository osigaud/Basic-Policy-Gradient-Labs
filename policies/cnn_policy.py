import torch
import torch.nn as nn
from policies.generic_net import GenericNet
from torch.distributions import Normal
import torch.nn.functional as func


class CnnPolicy(GenericNet):
    """
    Class used to represent a CNN policy
    Imported from github and never tested
    """
    def __init__(self, nb_actions, learning_rate):
        super(CnnPolicy, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=nb_actions)

        self.relu = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

    def forward(self, state):
        """
         Compute the pytorch tensors resulting from sending a state or vector of states through the policy network
         The obtained tensors can be used to obtain an action by calling select_action
         :param state: the input state(s)
         :return: the resulting pytorch tensor (here the max and standard deviation of a Gaussian probability of action)
         """
        state = self.relu(self.conv1(state))
        state = self.relu(self.conv2(state))
        state = self.relu(self.conv3(state))
        state = state.view(-1, 64 * 6 * 6)
        state = self.relu(self.fc1(state))
        scores = self.fc2(state)
        return func.softmax(scores, dim=1)

    # function to be verified
    def train_pg(self, state, action, reward):
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        mu, std = self.forward(state)
        loss = -Normal(mu, std).log_prob(action) * reward  # Negative score function x reward
        self.update(loss)
        return loss
