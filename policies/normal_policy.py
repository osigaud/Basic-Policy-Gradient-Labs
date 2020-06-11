import torch
import torch.nn as nn
from torch.distributions import Normal
from policies.generic_net import GenericNet

class NormalPolicy(GenericNet):
    def __init__(self, l1, l2, l3, l4, learning_rate):
        super(NormalPolicy, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc_mu = nn.Linear(l3, l4)
        self.fc_std = nn.Linear(l3, l4)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        mu = self.tanh(self.fc_mu(state))  # *2
        std = self.softplus(self.fc_std(state))  # + 1e-3
        return mu, std

    def select_action(self, state):
        with torch.no_grad():
            mu, std = self.forward(state)
            n = Normal(mu, std)
            action = n.sample()
        return action.data.numpy().astype(int)

    # function to be verified
    def train_pg(self, state, action, reward):
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)  # .unsqueeze(1)
        mu, std = self.forward(state)
        loss = -Normal(mu, std).log_prob(action) * reward  # Negative score function x reward
        self.update(loss)
        return loss

    def train_regress_from_batch(self, batch):
        for j in range(batch.size):
            episode = batch.episodes[j]
            state = np.array(episode.state_pool)
            action = np.array(episode.action_pool)
            self.train_regress(state, action)