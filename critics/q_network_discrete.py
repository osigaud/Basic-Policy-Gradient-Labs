import torch
import random
import torch.nn as nn
import torch.nn.functional as func
from critics.critic_network import CriticNetwork


class QNetworkDiscrete(CriticNetwork):
    """
    The kind of critic network for discrete actions used in DQN
    """
    def __init__(self, l1, l2, l3, l4, learning_rate):
        super(QNetworkDiscrete, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        """
          Compute the tensor value from a state, going through the network
          :param state: the given state(s)
          :return: the corresponding values, as a torch tensor
          """
        state = func.relu(self.fc1(state))
        state = func.relu(self.fc2(state))
        state = self.fc3(state)
        return state

    def sample_action(self, state, epsilon):
        """
        A specificity of discrete action critic networks: 
        an action can be chosen by taking the argmax over critic values
        This implementation includes exploration noise
        :param state: the input state
        :param epsilon: the exploration rate
        :return: the chosen action
        """
        out = self.forward(state)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()

    def train_bootstrap(self, q_target, state, action, reward, done, next_state, gamma):
        """
        Train the critic using a bootstrap method
        :param q_target: the target value
        :param state: the input state
        :param action: the input action
        :param reward: the obtained reward
        :param done: whether the step was final
        :param next_state: the obtained next state
        :param gamma: the discount factor
        :return: the obtained loss
        """
        q_out = self.forward(state)
        q_a = q_out.gather(1, action)
        max_q_prime = q_target(next_state).max(1)[0].unsqueeze(1)
        target = reward + gamma * max_q_prime * done
        loss = func.smooth_l1_loss(q_a, target)
        self.update(loss)
        return loss
