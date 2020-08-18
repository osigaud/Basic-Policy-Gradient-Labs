import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Bernoulli
from policies.generic_net import GenericNet


def make_det_vec(vals):
    retour = []
    for v in vals:
        if v > 0.5:
            retour.append(1.0)
        else:
            retour.append(0.0)
    return retour


class BernoulliPolicy(GenericNet):
    """
    A policy outputing a boolean value for each state
    """
    def __init__(self, l1, l2, l3, l4, learning_rate):
        super(BernoulliPolicy, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        # self.fc1.weight.data.uniform_(-1.0, 1.0)
        self.fc2 = nn.Linear(l2, l3)
        # self.fc2.weight.data.uniform_(-1.0, 1.0)
        self.fc3 = nn.Linear(l3, l4)  # Prob of Left
        # self.fc3.weight.data.uniform_(-1.0, 1.0)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        action = torch.sigmoid(self.fc3(state))
        return action

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            probs = self.forward(state)
            if deterministic:
                return make_det_vec(probs)
            else:
                action = Bernoulli(probs).sample()
            return action.data.numpy().astype(int)

    def train_pg(self, state, action, reward):
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        probs = self.forward(state)
        m = Bernoulli(probs)
        loss = -m.log_prob(action) * reward  # Negative score function x reward
        self.update(loss)
        return loss

    def train_regress(self, state, action):
        action = torch.FloatTensor(action)
        proposed_action = self.forward(state)
        loss = func.mse_loss(proposed_action, action)
        self.update(loss)
        return loss