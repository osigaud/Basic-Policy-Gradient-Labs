import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from policies import GenericNet


class QNet(GenericNet):
    def __init__(self, learning_rate, tau=0.01, state_size=3, action_size=1, hidden_layer1=128, hidden_layer2=32,
                 output_size=1):
        """Build a double Q network.

        Args:
            learning_rate (float): The learning rate.
            tau (float): The update rate between 0 and 1 used to update the target network.
            state_size (int): The number of elements in a single state vector.
            action_size (int): The number of elements in a single action vector.
            hidden_layer1 (int): The number of neurons of the first hidden layer (should be even).
            hidden_layer2 (int): The number of neurons of the second hidden layer.
            output_size (int): The number of outputs.
        """
        super(QNet, self).__init__()
        self.tau = tau

        half_hidden_layer1 = hidden_layer1 // 2
        self.fc_s = nn.Linear(state_size, half_hidden_layer1)
        self.fc_a = nn.Linear(action_size, half_hidden_layer1)
        self.fc_cat = nn.Linear(2 * half_hidden_layer1, hidden_layer2)
        self.fc_out = nn.Linear(hidden_layer2, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.losses = None

    def forward(self, state, action):
        """Forward the state and to action through the network.

        Args:
            state (torch.Tensor): The column tensor of state(s).
            action (torch.Tensor): The column tensor of action(s).

        Returns:
            torch.Tensor: The critic value at a state action pair.
        """
        h1 = F.relu(self.fc_s(state))
        h2 = F.relu(self.fc_a(action))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def evaluate(self, state, action):
        """Return the critic value at a state action pair, as a numpy structure

        Args:
            state (numpy.ndarray): the given state
            action (numpy.ndarray): the given action

        Returns:
             numpy.ndarray: The value.
        """
        state = torch.from_numpy(state).float().reshape(1, -1)
        action = torch.from_numpy(action).float().reshape(1, -1)
        x = self.forward(state, action)
        return x.data.numpy()

    def train_net(self, target, mini_batch):
        """Train the network.

        Args:
            target (torch.Tensor): The target value.
            mini_batch (torch.Tensor): The mini batch of data containing the state, the action, the reward,
            the next state and 1 - done in each row.
        """
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.losses = loss.data.numpy().astype(float).mean()
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, target_network):
        """Update the weights of the target network.

        Args:
            target_network (critics.QNet): The target network to update.
        """
        for param_target, param in zip(target_network.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
