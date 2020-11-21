import torch


class Simulation:
    def __init__(self, env, nb_episodes=400, update_threshold=1000, nb_updates=20, batch_size=32, print_interval=20):
        """Build a Simulation

        Args:
            env (gym.Env): the environment of the simulation
            nb_episodes (int): the number of episode to fully train the actor
            update_threshold (int): the minimum number of steps stored in the memory before training
            nb_updates (int): the number of updates to perform after each episode once the criterion is met
            batch_size (int): the size of the batch of data used to perform an update
            print_interval (int): the period in episodes to print the average reward over that period
        """
        self.env = env
        self.nb_episodes = nb_episodes
        self.update_threshold = update_threshold
        self.nb_updates = nb_updates
        self.batch_size = batch_size
        self.best_rew = -500
        self.print_interval = print_interval

    def _perform_episode(self, policy, memory=None):
        """Let the policy perform one episode.

        Each state, action, reward, done state and next state tuple can be stored in a memory.

        Args:
            policy (policies.PolicyNet): The policy used to choose an action.
            memory (memory.ReplayBuffer): The buffer used to store the tuples.

        Returns:
            int: The sum of the score of each step over the whole episode.
        """
        state = self.env.reset()
        is_done = False
        score = 0

        while not is_done:
            action, log_prob = policy.forward(torch.from_numpy(state).float())  # action selection
            next_state, reward, is_done, _ = self.env.step([2.0 * action.item()])
            # add of the global state in the replay buffer
            if memory is not None:
                memory.put((state, action.item(), reward / 10.0, next_state, is_done))
            score += reward
            state = next_state
        return score

    def _update_networks(self, memory, policy, critic):
        """Update the networks of the policy and the critic.

        Args:
            memory (memory.ReplayBuffer): The given buffer.
            policy (policies.PolicyNet): The given policy to update.
            critic (critics.DoubleQNet): The given critics to update.
        """
        mini_batch = memory.sample(self.batch_size)  # Extract a mini-batch from the memory
        td_target = critic.compute_target(policy, mini_batch)  # Compute the time differential target
        critic.train_net(td_target, mini_batch)  # Train both critic networks
        policy.train_net(critic, mini_batch)  # Train the actor

    def train(self, memory, policy_wrapper, critic, policy_loss_file, critic_loss_file):
        """Train the actor and the critic using the given memory.

        Args:
            memory (memory.ReplayBuffer): The given buffer.
            policy_wrapper (wrappers.PolicyWrapper): The wrapped policy to train.
            critic (critics.DoubleQNet): The critic to train.
            policy_loss_file: The opened file to store the loss of the policy
            critic_loss_file: The opened file to store the loss of the critic
        """
        score = 0.0
        policy = policy_wrapper.policy
        for episode in range(self.nb_episodes):
            score_episode = self._perform_episode(policy, memory)
            score += score_episode

            if memory.size() > self.update_threshold:
                for i in range(self.nb_updates):
                    self._update_networks(memory, policy, critic)

            if policy.losses is not None:
                policy_loss = policy.losses
                policy_loss_file.write(str(episode) + " " + str(policy_loss) + "\n")

            if critic.q1.losses is not None:
                critic1_loss = critic.q1.losses
                critic2_loss = critic.q2.losses
                critic_loss_file.write(str(episode) + " " + str(critic1_loss) + " " + str(critic2_loss) + "\n")

            if score_episode > self.best_rew:
                self.best_rew = score_episode
                policy_wrapper.save(self.best_rew)

            if episode % self.print_interval == 0 and episode != 0:
                print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(episode, score / self.print_interval,
                                                                                 policy.log_alpha.exp()))
                score = 0.0
