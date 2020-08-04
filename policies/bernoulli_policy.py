import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from policies.generic_net import GenericNet


def make_det_vec(vals):
    retour = []
    for v in vals:
        if v>0.5:
            retour.append(1.0)
        else:
            retour.append(0.0)
    return retour


class BernoulliPolicy(GenericNet):
    def __init__(self, l1, l2, l3, l4, learning_rate):
        super(BernoulliPolicy, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc3 = nn.Linear(l3, l4)  # Prob of Left
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        action = torch.sigmoid(self.fc3(state))
        return action

    def select_action(self, state):
        with torch.no_grad():
            probs = self.forward(state)
            action = Bernoulli(probs).sample()
        return action.data.numpy().astype(int)

    def select_action_deterministic(self, state):
        with torch.no_grad():
            probs = self.forward(state)
            vals = probs.data.numpy().astype(int)
        return make_det_vec(vals)

    def train_pg(self, state, action, reward):
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)  # .unsqueeze(1)
        probs = self.forward(state)
        m = Bernoulli(probs)
        loss = -m.log_prob(action) * reward  # Negative score function x reward
        self.update(loss)
        return loss

    def my_train_loop(self, state, action, reward):
        self.optimizer.zero_grad()
        # print(action)
        for i in range(len(action)):
            act = torch.FloatTensor(action[i])
            # reward = torch.FloatTensor(reward[i]).unsqueeze(1)
            probs = self.forward(state[i])
            m = Bernoulli(probs)
            loss = -m.log_prob(act) * reward[i]  # Negative score function x reward
            loss.backward()
        self.optimizer.step()
