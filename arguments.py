import argparse


# the following functions are used to build file names for saving data and displaying results


def make_study_string(params):
    return params.env_name + '_' + params.study_name + '_' + params.critic_update_method + '_' + \
           params.critic_estim_method


def make_study_params_string(params):
    return 'trajs_' + str(params.nb_trajs) + '_update_threshold_' + str(params.update_threshold) + '_nb_updates_' + str(
        params.nb_updates)


def make_learning_params_string(params):
    return 'gamma_' + str(params.gamma) + '_tau_' + str(params.tau) + '_nstep_' + str(params.nstep) + '_lr_act_' + str(
        params.lr_actor) + '_lr_critic_' + str(params.lr_critic) + '_init_alpha_' + str(
        params.init_alpha) + '_lr_alpha_' + str(params.lr_alpha) + '_target_entropy_alpha_' + str(
        params.target_entropy_alpha)


def make_full_string(params):
    return make_study_string(params) + '_' + make_study_params_string(params) + '_' + make_learning_params_string(
        params)


def get_args():
    """
    Standard function to specify the default value of the hyper-parameters of all policy gradient algorithms
    and experimental setups
    :return: the complete list of arguments
    """
    parser = argparse.ArgumentParser()
    # environment setting
    parser.add_argument('--env_name', type=str, default='Pendulum-v0', help='the environment name')
    parser.add_argument('--env_obs_space_name', nargs='+', type=str,
                        default=["pos", "angle"])  # ["pos", "angle", "vx", "v angle"]
    parser.add_argument('--render', type=bool, default=False, help='visualize the run or not')
    # study settings
    parser.add_argument('--study_name', type=str, default='pg', help='study name: pg, regress, nstep')
    parser.add_argument('--critic_update_method', type=str, default="dataset",
                        help='critic update method: batch or dataset')
    parser.add_argument('--policy_type', type=str, default="bernoulli",
                        help='policy type: bernoulli, normal, squashedGaussian, discrete')
    parser.add_argument('--team_name', type=str, default='default_team', help='team name')
    # study parameters
    parser.add_argument('--nb_repet', type=int, default=10, help='number of repetitions to get statistics')
    parser.add_argument('--nb_trajs', type=int, default=20, help='number of trajectories in a MC batch')
    parser.add_argument('--update_threshold', type=int, default=1000)
    parser.add_argument('--nb_updates', type=int, default=20, help='number of updates to the network per episode')
    parser.add_argument('--print_interval', type=int, default=20,
                        help='the period in episodes to print the average reward over that period')
    # algo settings
    parser.add_argument('--gradients', type=str, nargs='+', default=['sum', 'discount', 'normalize'],
                        help='other: baseline, beta')
    parser.add_argument('--critic_estim_method', type=str, default="td",
                        help='critic estimation method: mc, td or nstep')
    # learning parameters
    parser.add_argument('--batch_size', type=int, default=64, help='size of a minibatch')
    # Policy parameters
    parser.add_argument('--lr_actor', type=float, default=0.01, help='learning rate of the actor')
    parser.add_argument('--init_alpha', type=float, default=0.001)
    parser.add_argument('--lr_alpha', type=float, default=0.001)
    parser.add_argument('--target_entropy_alpha', type=float, default=-1.0)
    # Critic parameters
    parser.add_argument('--lr_critic', type=float, default=0.01, help='learning rate of the critic')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--tau', type=float, default=0.01)

    parser.add_argument('--beta', type=float, default=0.1, help='temperature in AWR-like learning')
    parser.add_argument('--nstep', type=int, default=5, help='n in n-step return')
    parser.add_argument('--max_episode_steps', type=int, default=None, help='duration of an episode (step limit)')

    args = parser.parse_args()
    return args
