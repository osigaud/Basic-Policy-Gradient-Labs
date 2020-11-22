import torch

from critics.q_net import QNet


class DoubleQNet:
    def __init__(self, learning_rate, gamma=0.98, tau=0.01):
        """Build a double Q network.

        Args:
            learning_rate (float): The learning rate for both Q networks.
            gamma (float): The reward decay.
            tau (float): The rate between 0 and 1 used to update the target networks.
        """
        self.gamma = gamma

        self.q1 = QNet(learning_rate, tau)
        self.q2 = QNet(learning_rate, tau)
        self.q1_target = QNet(learning_rate)
        self.q2_target = QNet(learning_rate)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def forward(self, state, action, using_target=False):
        """Forward the state and to action through the network.

        Args:
            state (torch.Tensor): The column tensor of state(s).
            action (torch.Tensor): The column tensor of action(s).
            using_target (bool): Whether to use the target networks or not.

        Returns:
            torch.Tensor: The minimum critic value at a state action pair of the two critics.
        """
        if using_target:
            q1_value = self.q1_target(state, action)
            q2_value = self.q2_target(state, action)
        else:
            q1_value = self.q1(state, action)
            q2_value = self.q2(state, action)
        q1_q2 = torch.cat([q1_value, q2_value], dim=1)
        return torch.min(q1_q2, 1, keepdim=True)[0]

    def evaluate(self, state, action):
        """Return the critic value at a state action pair, as a numpy structure.

        Args:
            state (numpy.ndarray): The given state.
            action (numpy.ndarray): The given action.
        Returns:
             numpy.ndarray: The value
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
        self.q1.train_net(target, mini_batch)
        self.q2.train_net(target, mini_batch)
        self.q1.soft_update(self.q1_target)
        self.q2.soft_update(self.q2_target)

    def compute_target(self, policy, mini_batch):
        """Compute the target based on the given policy and the mini batch.

        Args:
            policy (policies.PolicyNet): The given policy.
            mini_batch (torch.Tensor): The mini batch of data containing the state, the action, the reward,
            the next state and 1 - done in each row.

        Returns:
            torch.Tensor: The computed target.
        """
        state, action, reward, next_state, d = mini_batch

        with torch.no_grad():
            action_policy, log_prob = policy(next_state)
            entropy = -policy.log_alpha.exp() * log_prob
            min_q = self.forward(next_state, action_policy, using_target=True)
            target = reward + self.gamma * d * (min_q + entropy)

        return target
