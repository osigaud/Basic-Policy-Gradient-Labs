import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random
from policies.generic_net import GenericNet

#Hyperparameters
lr_pi           = 0.0005
lr_q            = 0.001
init_alpha     = 0.01
gamma           = 0.98
batch_size     = 32
buffer_limit   = 50000
tau             = 0.01 # for target network soft update
target_entropy = -1.0 # for automated alpha update
lr_alpha        = 0.001  # for automated alpha update

class SoftCritic(GenericNet):
    def __init__(self, l1, l2=16, l3= 16, l4=1, learning_rate=0.001):
        """ Initialize the q_network
        :param l1: number of neuron in the input layer
        :param l2: number of neuron in the first hidden layer
        :param l3: number of neuron in the second hidden layer 
        :param l4: number of neuron in the output layer
        :param learning_rate: learning coefficient
        """
        super(SoftCritic, self).__init__()
        self.fc_in = nn.Linear(l1,l2)   
        self.fc_h1 = nn.Linear(l2,l3)
        self.fc_out = nn.Linear(l3,l4)
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

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.update(loss)

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    def update(self, loss) -> None:
        """
        Apply a loss to a network using gradient backpropagation
        :param loss: the applied loss
        :return: nothing
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
