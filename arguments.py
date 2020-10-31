import argparse

# the following functions are used to build file names for saving data and displaying results


def make_study_string(params):
    return params.env_name + '_' + params.study_name + '_' + params.critic_update_method \
           + '_' + params.critic_estim_method+ '_eval_' + str(params.deterministic_eval)


def make_study_params_string(params):
    return 'cycles_' + str(params.nb_cycles) + '_trajs_' + str(params.nb_trajs) + '_batches_' + str(params.nb_batches)


def make_learning_params_string(params):
    return 'gamma_' + str(params.gamma) + '_nstep_' + str(params.nstep) + '_lr_act_' \
           + str(params.lr_actor) + '_lr_critic_' + str(params.lr_critic)


def make_full_string(params):
    return make_study_string(params) + '_' + make_study_params_string(params) + '_' \
           + make_learning_params_string(params)


def get_args():
    """
    Standard function to specify the default value of the hyper-parameters of all policy gradient algorithms
    and experimental setups
    :return: the complete list of arguments
    """
    parser = argparse.ArgumentParser()
    # environment setting
    parser.add_argument('--env_name', type=str, default='CartPoleContinuous-v0', help='the environment name')
    parser.add_argument('--env_obs_space_name', type=str, default=["pos", "angle"])  # ["pos", "angle", "vx", "v angle"]
    parser.add_argument('--render', type=bool, default=False, help='visualize the run or not')
    # study settings
    parser.add_argument('--study_name', type=str, default='pg', help='study name: pg, regress, nstep')
    parser.add_argument('--critic_update_method', type=str, default="dataset", help='critic update method: batch or dataset')
    parser.add_argument('--policy_type', type=str, default="bernoulli", help='policy type: bernoulli, normal, squashedGaussian, discrete')
    parser.add_argument('--team_name', type=str, default='default_team', help='team name')
    parser.add_argument('--deterministic_eval', type=bool, default=True, help='deterministic policy evaluation?')
    # study parameters
    parser.add_argument('--nb_repet', type=int, default=10, help='number of repetitions to get statistics')
    parser.add_argument('--nb_cycles', type=int, default=40, help='number of training cycles')
    parser.add_argument('--nb_trajs', type=int, default=20, help='number of trajectories in a MC batch')
    parser.add_argument('--nb_batches', type=int, default=20, help='number of updates of the network using datasets')
    # algo settings
    parser.add_argument('--gradients', type=str, nargs='+', default=['sum', 'discount', 'normalize'], help='other: baseline, beta')
    parser.add_argument('--critic_estim_method', type=str, default="td", help='critic estimation method: mc, td or nstep')
    # learning parameters
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lr_actor', type=float, default=0.01, help='learning rate of the actor')
    parser.add_argument('--lr_critic', type=float, default=0.01, help='learning rate of the critic')
    parser.add_argument('--beta', type=float, default=0.1, help='temperature in AWR-like learning')
    parser.add_argument('--nstep', type=int, default=5, help='n in n-step return')
    parser.add_argument('--batch_size', type=int, default=64, help='size of a minibatch')
    parser.add_argument('--nb_workers', type=int, default=2, help='number of cpus to collect samples')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle replay samples or not')
    parser.add_argument('--max_episode_steps', type=int, default=None, help='duration of an episode (step limit)')

    '''
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='experiments/saved_models/', help='the path to save the models')
    parser.add_argument('--save-files', type=str, default='experiments/saved_files/', help='the path to save the files')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replaced')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')


    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=100, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=1500, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=20, help='the rollouts per mpi')
    parser.add_argument('--use-curriculum', action='store_true', default=False, help='use the curriculum or not')
    parser.add_argument('--multiple-target-sizes', action='store_true', default=False, help='train the agent to reach targets of different sizes')
    parser.add_argument('--include-movement-cost', action='store_false', default=True, help='consider or not the cost of a movement')
    '''

    args = parser.parse_args()
    return args
