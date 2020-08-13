import numpy as np
import torch
import torch.utils.data as data
from episode import Episode


class Batch:
    def __init__(self):
        self.episodes = []
        self.size = 0

    def add_episode(self, episode):
        self.episodes.append(episode)
        self.size += 1

    def copy_batch(self):
        b2 = Batch()
        for i in range(self.size):
            ep = Episode()
            sep = self.episodes[i]
            for j in range(self.episodes[i].len):
                ep.add(sep.state_pool[j], sep.action_pool[j], sep.reward_pool[j],
                       sep.done_pool[j], sep.next_state_pool[j])
            b2.add_episode(ep)
        return b2

    def add_sample(self, state, action, reward, done, next_state):
        self.episodes[self.size].add(state, action, reward, done, next_state)

    def discounted_sum_rewards(self, gamma):
        for i in range(self.size):
            self.episodes[i].discounted_sum_rewards(gamma)

    def sum_rewards(self):
        for i in range(self.size):
            self.episodes[i].sum_rewards()

    def substract_baseline(self, critic):
        for i in range(self.size):
            self.episodes[i].substract_baseline(critic)

    def nstep_return(self, n, gamma, critic):
        for i in range(self.size):
            self.episodes[i].nstep_return(n, gamma, critic)

    def normalize_rewards(self, gamma):
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

    def exponentiate_rewards(self, beta):
        for i in range(self.size):
            self.episodes[i].exponentiate_rewards(beta)

    def train_td_actor(self, actor):
        losses = []
        for j in range(self.size):
            episode = self.episodes[j]
            state = np.array(episode.state_pool)
            action = np.array(episode.action_pool)
            reward = np.array(episode.reward_pool)
            value_loss = actor.train_pg(state, action, reward)
            loss = value_loss.data.numpy()
            mean_loss = loss.mean()
            losses.append(mean_loss)
        return np.array(losses).mean()

    def train_regress_actor(self, policy):
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

    def compute_td_critic(self, gamma, critic, actor, train):
        """
        
        :param gamma: 
        :param critic: 
        :param actor: 
        :param train: True to train, False to compute a validation loss
        :return: 
        """
        losses = []
        for j in range(self.size):
            episode = self.episodes[j]
            state = np.array(episode.state_pool)
            action = np.array(episode.action_pool)
            reward = np.array(episode.reward_pool)
            done = np.array(episode.done_pool)
            next_state = np.array(episode.next_state_pool)
            next_action = actor.select_action(next_state)
            target = critic.compute_bootstrap_target(reward, done, next_state, next_action, gamma)
            # print("s:", state.shape, "a:", action.shape, "r:", reward.shape, "d:", done.shape, "n:", next_state.shape)
            # print("r:",reward)
            target = torch.FloatTensor(target).unsqueeze(1)
            value_loss = critic.compute_target_loss(state, action, target, train)
            loss = value_loss.data.numpy()
            losses.append(loss)
        mean_loss = np.array(losses).mean()
        return mean_loss

    def compute_mc_critic(self, gamma, critic, n, train):
        """
        :param gamma: 
        :param critic: 
        :param n: 
        :param train: True to train, False to compute a validation loss
        :return: 
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
            # print("s:", state.shape, "a:", action.shape, "r:", reward.shape, "d:", done.shape, "n:", next_state.shape)
            # print("r:",reward)
            # print("t:", target)
            value_loss = critic.compute_target_loss(state, action, target, train)
            loss = value_loss.data.numpy()
            losses.append(loss)
        mean_loss = np.array(losses).mean()
        return mean_loss
        # losses.append(value_loss.mean().data.numpy())
        # return losses, targets

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
            target = episode.reward_pool
            list_targets = np.concatenate((list_targets, target))
            list_states = list_states + state
            list_actions = list_actions + action
            t_target = torch.Tensor(list_targets).unsqueeze(1)

        dataset = data.TensorDataset(torch.Tensor(list_states), torch.Tensor(list_actions), t_target)
        return dataset

    def prepare_dataset_td(self, params, actor, critic):
        """
        Computes the dataset of samples to allow for immediate update of the critic.
        The dataset contains the list of states, of actions, and the target value V(s) or Q(s,a)
        The computation of the target value depends on the critic update method.

        :param params: parameters
        :param actor: the actor, useful to determine the next action
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
            reward = episode.reward_pool
            if params.critic_estim_method == "td":
                done = np.array(episode.done_pool)
                next_state = np.array(episode.next_state_pool)
                next_action = actor.select_action(next_state)
                # print("s:", state.shape, "a:", action.shape, "r:", reward.shape, "d:", done.shape, "n:", next_state.shape)
                # print("r:",reward)
                # print("t:", target)
                target = critic.compute_bootstrap_target(reward, done, next_state, next_action, params.gamma)
            else:
                target = reward
            list_targets = np.concatenate((list_targets, target))
            list_states = list_states + state
            list_actions = list_actions + action
            t_target = torch.Tensor(list_targets).unsqueeze(1)

        dataset = data.TensorDataset(torch.Tensor(list_states), torch.Tensor(list_actions), t_target)
        return dataset


