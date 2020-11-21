import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from policies import GenericNet


class QNet(GenericNet):
    def __init__(self, learning_rate, gamma=0.98, tau=0.01):
        super(QNet, self).__init__()
        self.gamma = gamma
        self.tau = tau

        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.losses = None

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.losses = loss.data.numpy().astype(float).mean()
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def evaluate(self, state, action):
        """
        Return the critic value at a state action pair, as a numpy structure
        :param state: the given state
        :param action: the given action
        :return: the value
        """
        state = torch.from_numpy(state).float().reshape(1, -1)
        action = torch.from_numpy(action).float().reshape(1, -1)
        x = self.forward(state, action)
        return x.data.numpy()


def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + q1.gamma * done * (min_q + entropy)

    return target
