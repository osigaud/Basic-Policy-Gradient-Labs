import numpy as np
from itertools import count
from batch import Episode, Batch
from environment import make_env


def make_simu_from_wrapper(pw, params):
    env_name = pw.env_name
    params.env_name = env_name
    env, discrete = make_env(env_name, params.policy_type, params.env_obs_space_name)
    if params.max_episode_steps is not None:
        env._max_episode_steps = params.max_episode_steps
    return Simu(env, env_name, discrete)


def make_simu_from_params(params):
    env_name = params.env_name
    env, discrete = make_env(env_name, params.policy_type, params.env_obs_space_name)
    if params.max_episode_steps is not None:
        env._max_episode_steps = params.max_episode_steps
    return Simu(env, env_name, discrete)


class Simu:
    def __init__(self, env, name, discrete):
        self.cpt = 0
        self.best_reward = -1e38
        self.env = env
        self.name = name
        self.discrete = discrete
        self.obs_size = env.observation_space.shape[0]

    def reset(self, render):
        state = self.env.reset()
        if render:
            self.env.render(mode='rgb_array')
        return state

    def take_step(self, state, action, episode, render=False):
        next_state, reward, done, _ = self.env.step(action)
        if render:
            self.env.render(mode='rgb_array')
        episode.add(state, action, reward, done, next_state)
        return next_state, reward, done

    # used to evaluate an already trained policy, without training nor storing data
    def evaluate_episode(self, policy, render=False):
        state = self.reset(render)

        total_reward = 0

        for _ in count():
            action = policy.select_action_deterministic(state)
            print(action)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state

            if done:
                return total_reward

    def perform_one_episode(self, policy, render):
        state = self.reset(render)
        total_reward = 0
        episode = Episode()
        for _ in count():
            action = policy.select_action(state)
            state, reward, done = self.take_step(state, action, episode, render)
            total_reward += reward

            if done:
                # print(batch.episodes[0].action_pool)
                return episode, total_reward

    def perform_episodes(self, pw, params, policy, critic, policy_loss_file, value_loss_file, study_name, beta=0) -> None:
        batch = Batch()
        for cycle in range(params.nb_cycles):
            total_reward = np.zeros(params.nb_evals)
            for e in range(params.nb_evals):
                episode, total_reward[e] = self.perform_one_episode(policy, params.render)
                batch.add_episode(episode)

            average_reward = np.mean(total_reward)
            if self.best_reward < average_reward:
                self.best_reward = average_reward-1
                pw.save(self.best_reward)

            # Update policy
            batch2 = batch.copy_batch()
            batch.prepare(params.gamma, beta, critic, study_name, params.nstep)
            policy_loss = batch.train_td_actor(policy)

            if params.critic_update == "dataset":
                if params.critic_estim == "td":
                    dataset = batch2.prepare_dataset_td(params, policy, critic)
                    # value_loss = update_critic_valid_td(params, dataset, critic)
                    value_loss = critic.update_td(params, dataset, True)
                elif params.critic_estim == "mc":
                    dataset = batch2.prepare_dataset_mc(params.gamma)
                    # value_loss = update_critic_valid_mc(params, dataset, critic, save_best=False)
                    value_loss = critic.update_mc(params, dataset, value_loss_file, True, save_best=False)
                elif params.critic_estim == "nstep":
                    dataset = batch2.prepare_dataset_td(params.gamma, policy, critic, "nstep")
                    value_loss = critic.update_valid_td(params, dataset)
                else:
                    print("perform episodes : critic estim method inconnue : ", params.critic_estim)
            elif params.critic_update == "batch":
                if params.critic_estim == "td":
                    value_loss = batch2.compute_td_critic(params.gamma, critic, policy, True)
                elif params.critic_estim == "mc":
                    value_loss = batch2.compute_mc_critic(params.gamma, critic, 0, True)
                elif params.critic_estim == "nstep":
                    value_loss = batch2.compute_mc_critic(params.gamma, critic, params.nstep, True)
                else:
                    print("perform episodes : critic estim method inconnue : ", params.critic_estim)
            else:
                print("perform episodes : update method inconnue : ", params.critic_update)
            value_loss_file.write(str(cycle) + " " + str(value_loss) + "\n")
            policy_loss_file.write(str(cycle) + " " + str(policy_loss) + "\n")
            batch = Batch()

    def make_monte_carlo_batch(self, params, policy):
        batch = Batch()
        self.env.set_reward_flag(False)
        self.env.set_duration_flag(False)
        for e in range(params.nb_evals):
            episode, _ = self.perform_one_episode(policy, params.render)
            batch.add_episode(episode)
        return batch
