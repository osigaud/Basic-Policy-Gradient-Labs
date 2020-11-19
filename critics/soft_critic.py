import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from policies.generic_net import GenericNet


class SoftCritic(GenericNet):
    def __init__(self, politic, l1, l2=16, l3=16, l4=1, learning_rate=0.001):
        """ Initialize the q_network
        :param l1: number of neuron in the input layer
        :param l2: number of neuron in the first hidden layer
        :param l3: number of neuron in the second hidden layer
        :param l4: number of neuron in the output layer
        :param learning_rate: learning coefficient
        """
        super(SoftCritic, self).__init__()
        self.politic = politic
        self.q1 = QNet(learning_rate)
        self.q1_target = QNet(learning_rate)
        self.q2 = QNet(learning_rate)
        self.q2_target = QNet(learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state, action):
        """
         Compute the value from a state action pair, going through the network
         :param action: the chosen action
         :param state: the given state(s)
         :return: the corresponding values, as a torch tensor
         """
        action = np.reshape(action, (-1, 1))
        if state.ndim == 1:
            state = np.reshape(state, (1, -1))
        x = np.hstack((state, action))
        state_action = torch.from_numpy(x).float()
        h1 = F.relu(self.fc_in(state_action))
        h2 = F.relu(self.fc_h1(h1))
        q = self.fc_out(h2)
        return q

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
        with torch.no_grad():
            a_prime, log_prob = self.politic(next_state)
            entropy = -self.politic.log_alpha.exp() * log_prob
            q1_val = self.q1_target(next_state, a_prime)
            q2_val = self.q2_target(next_state, a_prime)
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = reward + gamma * (1 - done) * (min_q + entropy)

            return target

    def compute_loss_to_target(self, state, action, target):
        """
        Compute the MSE between a target value and the critic value for the state action pair(s)
        :param state: a state or vector of state
        :param action: an action or vector of actions
        :param target: the target value
        :return: the resulting loss
        """
        return F.smooth_l1_loss(self.forward(state, action), target)

    def update(self, loss):
        """
        Apply a loss to a network using gradient backpropagation
        :param loss: the applied loss
        :return: nothing
        """
        self.q1.update(loss[:, 0])
        self.q2.update(loss[:, 1])
        mini_batch = memory.sample(batch_size)
        td_target = calc_target(pi, q1_target, q2_target, mini_batch)
        q1.train_net(td_target, mini_batch)
        q2.train_net(td_target, mini_batch)
        entropy = pi.train_net(q1, q2, mini_batch)
        q1.soft_update(q1_target)
        q2.soft_update(q2_target)


class QNet(nn.Module):
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def soft_update(self, net_target):
        for param_target, param in zip(
                net_target.parameters(),
                self.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - tau) + param.data * tau)

    def compute_loss_to_target(self, state, action, target):
        """
        Compute the MSE between a target value and the critic value for the state action pair(s)
        :param state: a state or vector of state
        :param action: an action or vector of actions
        :param target: the target value
        :return: the resulting loss
        """
        return F.smooth_l1_loss(self.forward(state, action), target)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
