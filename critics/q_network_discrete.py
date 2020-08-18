import torch
import random
import torch.nn as nn
import torch.nn.functional as func
from critics.critic_network import CriticNetwork


class QNetworkDiscrete(CriticNetwork):
    def __init__(self, l1, l2, l3, l4, learning_rate):
        super(QNetworkDiscrete, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()

    def train_bootstrap(self, q_target, state, action, reward, done, next_state, gamma):
        q_out = self.forward(state)
        q_a = q_out.gather(1, action)
        max_q_prime = q_target(next_state).max(1)[0].unsqueeze(1)
        target = reward + gamma * max_q_prime * done
        loss = func.smooth_l1_loss(q_a, target)
        self.update(loss)
