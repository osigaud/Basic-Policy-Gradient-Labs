import numpy as np
import torch
import torch.utils.data as data
from episode import Episode


class Batch:
    """
    A batch of samples, collected into a vector of episode
    """
    def __init__(self):
        self.episodes = []
        self.size = 0

    def add_episode(self, episode) -> None:
        """
        Ad an episode to the batch
        :param episode: the added episode
        :return: nothing
        """
        self.episodes.append(episode)
        self.size += 1

    def copy_batch(self):
        """
        Make a copy of the current batch
        :return: the copied batch
        """
        b2 = Batch()
        for i in range(self.size):
            ep = Episode()
            sep = self.episodes[i]
            for j in range(self.episodes[i].len):
                ep.add(sep.state_pool[j], sep.action_pool[j], sep.reward_pool[j],
                       sep.done_pool[j], sep.next_state_pool[j])
            b2.add_episode(ep)
        return b2

    def add_sample(self, state, action, reward, done, next_state) -> None:
        """
        Add a sample to the current episode
        :param state: the current state
        :param action: the taken action
        :param reward: the resulting reward
        :param done: whether the episode is over
        :param next_state: the resulting next state
        :return: nothing
        """
        self.episodes[self.size].add(state, action, reward, done, next_state)

    def discounted_sum_rewards(self, gamma) -> None:
        """
        Apply a discounted sum of rewards to all samples of all episodes
        :param gamma: the discount factor
        :return: nothing
        """
        for i in range(self.size):
            self.episodes[i].discounted_sum_rewards(gamma)

    def sum_rewards(self) -> None:
        """
        Apply a sum of rewards to all samples of all episodes
        :return: nothing
        """
        for i in range(self.size):
            self.episodes[i].sum_rewards()

    def substract_baseline(self, critic) -> None:
        """
        Substracts a baseline to the reward of all samples of all episodes
        :param critic: the baseline critic to be substracted
        :return: nothing
        """
        for i in range(self.size):
            self.episodes[i].substract_baseline(critic)

    def nstep_return(self, n, gamma, critic) -> None:
        """
        Apply Bellman backup n-step return to all rewards of all samples of all episodes
        :param n: the number of steps in n-step
        :param gamma: the discount factor
        :param critic: the critic used to perform Bellman backups
        :return: nothing
        """
        for i in range(self.size):
            self.episodes[i].nstep_return(n, gamma, critic)

    def normalize_rewards(self, gamma) -> None:
        """
         Apply a normalized and discounted sum of rewards to all samples of all episodes
         :param gamma: the discount factor
         :return: nothing
         """
        reward_pool = []
        for i in range(self.size):
            self.episodes[i].discounted_sum_rewards(gamma)
            reward_pool += self.episodes[i].reward_pool
        reward_std = np.std(reward_pool)
        if reward_std > 0:
            reward_mean = np.mean(reward_pool)
            # print("normalize_rewards : ", reward_std, "mean=", reward_mean)
            for i in range(self.size):
                self.episodes[i].normalize_discounted_rewards(gamma, reward_mean, reward_std)
        else:
            reward_mean = np.mean(reward_pool)
            print("normalize_rewards : std=0, mean=", reward_mean)
            for i in range(self.size):
                self.episodes[i].normalize_discounted_rewards(gamma, reward_mean, 1.0)

    def exponentiate_rewards(self, beta) -> None:
        """
        Apply an exponentiation factor to the rewards of all samples of all episodes
        :param beta: the exponentiation factor
        :return: nothing
        """
        for i in range(self.size):
            self.episodes[i].exponentiate_rewards(beta)

    def train_policy_td(self, policy):
        """
        Trains a policy through a temporal difference method from a batch of data
        :param policy: the trained policy
        :return: the average loss over the batch
        """
        do_print = False
        losses = []
        if do_print: print("training data :")
        for j in range(self.size):
            episode = self.episodes[j]
            state = np.array(episode.state_pool)
            action = np.array(episode.action_pool)
            reward = np.array(episode.reward_pool)
            if do_print: print("state", state)
            if do_print: print("action", action)
            if do_print: print("reward", reward)
            policy_loss = policy.train_pg(state, action, reward)
            if do_print: print("loss", policy_loss)
            policy_loss = policy_loss.data.numpy()
            mean_loss = policy_loss.mean()
            losses.append(mean_loss)
        if do_print: print("end of training data :")
        return np.array(losses).mean()

    def train_policy_through_regress(self, policy):
        """
        Trains a policy through regression from a batch of data
        Moves the policy closer to performing the same action in the same state
        :param policy: the trained policy
        :return: the average loss over the batch
        """
        losses = []
        for j in range(self.size):
            episode = self.episodes[j]
            state = np.array(episode.state_pool)
            action = np.array(episode.action_pool)
            policy_loss = policy.train_regress(state, action)
            loss = policy_loss.data.numpy()
            mean_loss = loss.mean()
            losses.append(mean_loss)
        return np.array(losses).mean()

    def train_critic_td(self, gamma, policy, critic, train):
        """
        Trains a critic through a temporal difference method
        :param gamma: the discount factor
        :param critic: the trained critic
        :param policy: 
        :param train: True to train, False to compute a validation loss
        :return: the average critic loss
        """
        losses = []
        for j in range(self.size):
            episode = self.episodes[j]
            state = np.array(episode.state_pool)
            action = np.array(episode.action_pool)
            reward = np.array(episode.reward_pool)
            done = np.array(episode.done_pool)
            next_state = np.array(episode.next_state_pool)
            next_action = policy.select_action(next_state)
            target = critic.compute_bootstrap_target(reward, done, next_state, next_action, gamma)
            target = torch.FloatTensor(target).unsqueeze(1)
            critic_loss = critic.compute_loss_to_target(state, action, target)
            if train:
                critic.update(critic_loss)
            critic_loss = critic_loss.data.numpy()
            losses.append(critic_loss)
        mean_loss = np.array(losses).mean()
        return mean_loss

    def train_critic_mc(self, gamma, critic, n, train):
        """
        Trains a critic through a Monte Carlo method. Also used to perform n-step training
        :param gamma: the discount factor
        :param critic: the trained critic
        :param n: the n in n-step training
        :param train: True to train, False to just compute a validation loss
        :return: the average critic loss
        """
        if n == 0:
            self.discounted_sum_rewards(gamma)
        else:
            self.nstep_return(n, gamma, critic)
        losses = []
        targets = []
        for j in range(self.size):
            episode = self.episodes[j]
            state = np.array(episode.state_pool)
            action = np.array(episode.action_pool)
            reward = np.array(episode.reward_pool)
            target = torch.FloatTensor(reward).unsqueeze(1)
            targets.append(target.mean().data.numpy())
            critic_loss = critic.compute_loss_to_target(state, action, target)
            if train:
                critic.update(critic_loss)
            critic_loss = critic_loss.data.numpy()
            losses.append(critic_loss)
        mean_loss = np.array(losses).mean()
        return mean_loss

    def prepare_dataset_mc(self, gamma):
        """
        Computes the dataset of samples to allow for immediate update of the critic.
        The dataset contains the list of states, of actions, and the target value V(s) or Q(s,a)
        The computation of the target value depends on the critic update method.
        
        :param gamma: the discount factor
        :return: the dataset corresponding to the content of the replay buffer
        """
        list_targets = []
        list_states = []
        list_actions = []

        # prepare reward data for the mc case
        self.discounted_sum_rewards(gamma)

        for j in range(self.size):
            episode = self.episodes[j]
            state = episode.state_pool
            action = episode.action_pool
            #### MODIF:  transform actions in array, without it the dataset conversion in TensorDataset will crash on MountainCar and CartPole
            action_cp = []
            for i in range(len(action)) : 
                action_cp.append([int(action[i])])
            action = action_cp
            ####
            target = episode.reward_pool
            list_targets = np.concatenate((list_targets, target))
            list_states = list_states + state
            list_actions = list_actions + action
            t_target = torch.Tensor(list_targets).unsqueeze(1)

        dataset = data.TensorDataset(torch.Tensor(list_states), torch.Tensor(list_actions), t_target)
        return dataset

    def prepare_dataset_td(self, params, policy, critic):
        """
        Computes the dataset of samples to allow for immediate update of the critic.
        The dataset contains the list of states, of actions, and the target value V(s) or Q(s,a)
        The computation of the target value depends on the critic update method.

        :param params: parameters
        :param policy: the actor, useful to determine the next action
        :param critic: the critic to be updated (useful to compute the target value)

        :return: the dataset corresponding to the content of the replay buffer
        """
        list_targets = []
        list_states = []
        list_actions = []

        # prepare reward data for the td and n-step case
        if params.critic_estim_method == "nstep":
            self.nstep_return(params.nstep, params.gamma, critic)
        else:
            if not params.critic_estim_method == "td":
                print("batch prepare_dataset_td: unknown estimation method :", params.critic_estim_method)

        for j in range(self.size):
            episode = self.episodes[j]
            state = episode.state_pool
            action = episode.action_pool
            #### MODIF:  transform actions in array, without it the dataset conversion in TensorDataset will crash on MountainCar and CartPole
            action_cp = []
            for i in range(len(action)) : 
                action_cp.append([int(action[i])])
            action = action_cp
            ####
            reward = episode.reward_pool
            if params.critic_estim_method == "td":
                done = np.array(episode.done_pool)
                next_state = np.array(episode.next_state_pool)
                next_action = policy.select_action(next_state)
                target = critic.compute_bootstrap_target(reward, done, next_state, next_action, params.gamma)
            else:
                target = reward
            list_targets = np.concatenate((list_targets, target))
            list_states = list_states + state
            list_actions = list_actions + action
            t_target = torch.Tensor(list_targets).unsqueeze(1)

        dataset = data.TensorDataset(torch.Tensor(list_states), torch.Tensor(list_actions), t_target)
        return dataset


