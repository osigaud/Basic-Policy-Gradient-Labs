import torch

from critics.q_net import calc_target


class Simulation:
    def __init__(self, env, nb_episodes=400, update_threshold=1000, nb_updates=20, batch_size=32, print_interval=20):
        self.env = env
        self.nb_episodes = nb_episodes
        self.update_threshold = update_threshold
        self.nb_updates = nb_updates
        self.batch_size = batch_size
        self.best_rew = -1e30
        self.print_interval = print_interval

    def _perform_episode(self, policy, memory):
        s = self.env.reset()
        done = False
        score = 0

        while not done:
            a, log_prob = policy.forward(torch.from_numpy(s).float())  # action selection
            s_prime, r, done, info = self.env.step([2.0 * a.item()])
            # add of the global state in the replay buffer
            memory.put((s, a.item(), r / 10.0, s_prime, done))
            score += r
            s = s_prime
        return score

    def _update_networks(self, memory, policy, q1, q2, q1_target, q2_target):
        # Extract a mini-batch from the memory
        mini_batch = memory.sample(self.batch_size)
        # Compute the time differential target
        td_target = calc_target(policy, q1_target, q2_target, mini_batch)
        # Train both critic networks
        q1.train_net(td_target, mini_batch)
        q2.train_net(td_target, mini_batch)
        # Train the actor
        policy.train_net(q1, q2, mini_batch)
        # Update the weights of the target critic networks
        q1.soft_update(q1_target)
        q2.soft_update(q2_target)

    def train(self, memory, policy_wrapper, q1, q2, q1_target, q2_target, policy_loss_file, critic_loss_file):
        score = 0.0
        policy = policy_wrapper.policy
        for n_epi in range(self.nb_episodes):
            score_epi = self._perform_episode(policy, memory)
            score += score_epi

            if memory.size() > self.update_threshold:
                for i in range(self.nb_updates):
                    self._update_networks(memory, policy, q1, q2, q1_target, q2_target)

            if policy.losses is not None:
                policy_loss = policy.losses
                policy_loss_file.write(str(n_epi) + " " + str(policy_loss) + "\n")

            if q1.losses is not None:
                critic_loss = q1.losses
                critic_loss_file.write(str(n_epi) + " " + str(critic_loss) + "\n")

            if score_epi > self.best_rew:
                self.best_rew = score_epi
                policy_wrapper.save(self.best_rew)

            if n_epi % self.print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, score / self.print_interval,
                    policy.log_alpha.exp()))
                score = 0.0
