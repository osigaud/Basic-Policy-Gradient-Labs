import numpy as np
from itertools import count
from batch import Episode, Batch
from environment import make_env
from algo import Algo
from visu.visu_trajectories import plot_trajectory
from visu.visu_weights import plot_weight_histograms, plot_normal_histograms


def make_simu_from_params(params):
    """
    Creates the environment, adding the required wrappers
    :param params: the hyper-parameters of the run, specified in arguments.py or in the command line
    :return: a simulation object
    """
    env_name = params.env_name
    env = make_env(env_name, params.policy_type, params.max_episode_steps, params.env_obs_space_name)
    return Simu(env, env_name)


def make_simu_from_wrapper(pw, params):
    """
    Creates the environment, adding the required wrappers
    Used when loading an agent from an external file, through a policy wrapper
    :param pw: the policy wrapper specifying the environment
    :param params: the hyper-parameters of the run, specified in arguments.py or in the command line
    :return: a simulation object
    """
    env_name = pw.env_name
    params.env_name = env_name
    env = make_env(env_name, params.policy_type, params.max_episode_steps, params.env_obs_space_name)
    return Simu(env, env_name)


class Simu:
    """
    The class implements the interaction between the agent represented by its policy and the environment
    """
    def __init__(self, env, env_name):
        self.cpt = 0
        self.best_reward = -1e38
        self.env = env
        self.env_name = env_name
        self.obs_size = env.observation_space.shape[0]
        self.discrete = not env.action_space.contains(np.array([0.5]))

    def reset(self, render):
        """
        Reinitialize the state of the agent in the environment
        :param render: whether the step is displayed or not (True or False)
        :return: the new state
        """
        state = self.env.reset()
        if render:
            self.env.render(mode='rgb_array')
        return state

    def take_step(self, state, action, episode, render=False):
        """
        Perform one interaction step between the agent and the environment
        :param state: the current state of the agent
        :param action: the action selected by the agent
        :param episode: the structure into which the resulting sample will be added
        :param render: whether the step is displayed or not (True or False)
        :return: the resulting next state, reward, and whether the episode ended
        """
        next_state, reward, done, _ = self.env.step(action)
        if render:
            self.env.render(mode='rgb_array')
        episode.add(state, action, reward, done, next_state)
        return next_state, reward, done

    def evaluate_episode(self, policy, deterministic, render=False):
        """
         Perform an episode using the policy parameter and return the obtained reward
         Used to evaluate an already trained policy, without storing data for further training
         :param policy: the policy controlling the agent
         :param deterministic: whether the evaluation should use a deterministic policy or not
         :param render: whether the episode is displayed or not (True or False)
         :return: the total reward collected during the episode
         """
        self.env.set_reward_flag(True)
        self.env.set_duration_flag(True)
        state = self.reset(render)
        total_reward = 0
        for t in count():
            action = policy.select_action(state, deterministic)
            # print("action", action)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state

            if done:
                # print("############ eval nb steps:", t)
                return total_reward

    def train_on_one_episode(self, policy, deterministic, render):
        """
        Perform an episode using the policy parameter and return the corresponding samples into an episode structure
        :param policy: the policy controlling the agent
        :param deterministic: whether the evaluation should use a deterministic policy or not
        :param render: whether the episode is displayed or not (True or False)
        :return: the samples stored into an episode
        """
        state = self.reset(render)
        episode = Episode()
        for t in count():
            action = policy.select_action(state, deterministic)
            next_state, _, done = self.take_step(state, action, episode, render)
            state = next_state

            if done:
                # print("train nb steps:", t)
                return episode

    def train(self, pw, params, policy, critic, policy_loss_file, critic_loss_file, study_name, beta=0) -> None:
        """
        The main function for training and evaluating a policy
        Repeats training and evaluation params.nb_cycles times
        Stores the value and policy losses at each cycle
        When the reward is greater than the best reward so far, saves the corresponding policy
        :param pw: a policy wrapper, used to save the best policy into a file
        :param params: the hyper-parameters of the run, specified in arguments.py or in the command line
        :param policy: the trained policy
        :param critic: the corresponding critic (not always used)
        :param policy_loss_file: the file to record successive policy loss values
        :param critic_loss_file: the file to record successive critic loss values
        :param study_name: the name of the studied gradient algorithm
        :param beta: a specific parameter for beta-parametrized values
        :return: nothing
        """
        for cycle in range(params.nb_cycles):
            batch = self.make_monte_carlo_batch(params.nb_trajs, params.render, policy)

            # Update the policy
            batch2 = batch.copy_batch()
            algo = Algo(study_name, params.critic_estim_method, policy, critic, params.gamma, beta, params.nstep)
            algo.prepare_batch(batch)
            policy_loss = batch.train_policy_td(policy)

            # Update the critic
            assert params.critic_update_method in ['batch', 'dataset'], 'unsupported critic update method'
            if params.critic_update_method == "dataset":
                critic_loss = algo.train_critic_from_dataset(batch2, params)
            elif params.critic_update_method == "batch":
                critic_loss = algo.train_critic_from_batch(batch2)
            critic_loss_file.write(str(cycle) + " " + str(critic_loss) + "\n")
            policy_loss_file.write(str(cycle) + " " + str(policy_loss) + "\n")

            # policy evaluation part
            total_reward = self.evaluate_episode(policy, params.deterministic_eval)
            # plot_trajectory(batch2, self.env, cycle+1)

            # save best reward agent (no need for averaging if the policy is deterministic)
            if self.best_reward < total_reward:
                self.best_reward = total_reward
                pw.save(self.best_reward)

    def make_monte_carlo_batch(self, nb_episodes, render, policy):
        """
        Create a batch of episodes with a given policy
        Used in Monte Carlo approaches
        :param nb_episodes: the number of episodes in the batch
        :param render: whether the episode is displayed or not (True or False)
        :param policy: the policy controlling the agent
        :return: the resulting batch of episodes
        """
        batch = Batch()
        self.env.set_reward_flag(False)
        self.env.set_duration_flag(False)
        for e in range(nb_episodes):
            episode = self.train_on_one_episode(policy, False, render)
            batch.add_episode(episode)
        return batch
