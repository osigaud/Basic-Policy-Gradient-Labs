import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Bernoulli
from policies.generic_net import GenericNet


def make_det_vec(values):
    """
    Transform the output vector of a Bernoulli policy into a vector of deterministic choices
    :param values: the Bernoulli policy output vector (turned into a numpy array)
    :return: the vector of binary choices
    """
    choices = []
    for v in values:
        if v > 0.5:
            choices.append(1.0)
        else:
            choices.append(0.0)
    return choices


class BernoulliPolicy(GenericNet):
    """
    A policy whose probabilistic output is a boolean value for each state
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        """
         Compute the pytorch tensors resulting from sending a state or vector of states through the policy network
         The obtained tensors can be used to obtain an action by calling select_action
         :param state: the input state(s)
         :return: the resulting pytorch tensor (here the probability of giving 0 or 1 as output)
         """
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        action = torch.sigmoid(self.fc3(state))
        return action

    def select_action(self, state, deterministic=False):
        """
        Compute an action or vector of actions given a state or vector of states
        :param state: the input state(s)
        :param deterministic: whether the policy should be considered deterministic or not
        :return: the resulting action(s)
        """
        with torch.no_grad():
            probs = self.forward(state)
            if deterministic:
                return make_det_vec(probs)
            else:
                action = Bernoulli(probs).sample()
            return action.data.numpy().astype(int)

    def train_pg(self, state, action, reward):
        """
        Train the policy using a policy gradient approach
        :param state: the input state(s)
        :param action: the input action(s)
        :param reward: the resulting reward
        :return: the loss applied to train the policy
        """
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        probs = self.forward(state)
        m = Bernoulli(probs)
        loss = -m.log_prob(action) * reward  # Negative score function x reward
        self.update(loss)
        return loss

    def train_regress(self, state, action):
        """
          Train the policy to perform the same action(s) in the same state(s) using regression
          :param state: the input state(s)
          :param action: the input action(s)
          :return: the loss applied to train the policy
          """
        action = torch.FloatTensor(action)
        proposed_action = self.forward(state)
        loss = func.mse_loss(proposed_action, action)
        self.update(loss)
        return loss
