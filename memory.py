import collections
import random

import torch


class ReplayBuffer():
    def __init__(self, limit=50_000):
        """Build a replay buffer.

        Stores data from each step and acts as a FIFO when the limit is reached.

        Args:
            limit (int): The maximum length of the buffer.
        """
        self.buffer = collections.deque(maxlen=limit)

    def put(self, transition):
        """Stores a tuple.

        The tuple should contain:

        1. The state as a torch.Tensor
        2. The action as a float
        3. The rescaled reward as a float
        4. The following state as torch.Tensor
        5. 0 if done otherwise 1

        Args:
            transition (tuple): The transition to store.
        """
        self.buffer.append(transition)

    def sample(self, n):
        """Randomly extracts a sample of data containing n transitions.

        Args:
            n (int): The length of the mini batch.

        Returns:
            torch.Tensor: The mini batch
        """
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), torch.tensor(r_lst,
                                                                                                            dtype=torch.float), torch.tensor(
            s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        """Gets the length.

        Returns:
            int: The length.
        """
        return len(self.buffer)
