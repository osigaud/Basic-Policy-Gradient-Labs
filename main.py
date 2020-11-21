import os

from arguments import get_args
from chrono import Chrono
from critics.q_net import QNet
from environment import make_env
from memory import ReplayBuffer
from policies.policy_net import PolicyNet
from simulation import Simulation
from visu.visu_critics import plot_critic
from visu.visu_policies import plot_policy
from visu.visu_results import plot_results
from wrappers.policy_wrapper import PolicyWrapper


def create_data_folders() -> None:
    """
    Create folders where to save output files if they are not already there
    :return: nothing
    """
    if not os.path.exists("data/"):
        os.mkdir("./data")
    if not os.path.exists("data/save"):
        os.mkdir("./data/save")
    if not os.path.exists("data/critics"):
        os.mkdir("./data/critics")
    if not os.path.exists('data/policies/'):
        os.mkdir('data/policies/')
    if not os.path.exists('data/results/'):
        os.mkdir('data/results/')


def set_files(study_name, env_name):
    """
    Create files to save the policy loss and the critic loss
    :param study_name: the name of the study
    :param env_name: the name of the environment
    :return:
    """
    policy_loss_name = "data/save/policy_loss_" + study_name + '_' + env_name + ".txt"
    policy_loss_file = open(policy_loss_name, "w")
    critic_loss_name = "data/save/critic_loss_" + study_name + '_' + env_name + ".txt"
    critic_loss_file = open(critic_loss_name, "w")
    return policy_loss_file, critic_loss_file


def main(params) -> None:
    lr_pi = 0.0005  # params.lr_policy
    lr_q = 0.001  # params.lr_critics
    env_name = 'Pendulum-v0'  # params.env_name

    env = make_env(env_name, 'sac', params.max_episode_steps, params.env_obs_space_name)
    env.set_file_name('SAC' + '_' + env_name)

    simulation = Simulation(env)

    policy_loss_file, critic_loss_file = set_files('SAC', env_name)

    chrono = Chrono()

    for j in range(params.nb_repet):
        env.reinit()
        memory = ReplayBuffer()

        # Initialise the policy/actor
        policy = PolicyNet(lr_pi, init_alpha=0.02)
        pw = PolicyWrapper(policy, params.policy_type, env_name, params.team_name, params.max_episode_steps)
        # Initialise the critics
        q1 = QNet(lr_q)
        q2 = QNet(lr_q)
        q1_target = QNet(lr_q)
        q2_target = QNet(lr_q)
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())

        plot_policy(policy, env, True, env_name, 'SAC', '_ante_', j, plot=False)

        simulation.train(memory, pw, q1, q2, q1_target, q2_target, policy_loss_file, critic_loss_file)

        plot_policy(policy, env, True, env_name, 'SAC', '_post_', j, plot=False)
        plot_critic(env, env_name, q1, policy, 'SAC', '_q1_post_', j)
        plot_critic(env, env_name, q2, policy, 'SAC', '_q2_post_', j)
        q1.save_model('data/critics/{}#{}#SAC{}.pt'.format(params.env_name, params.team_name, str(j)))
        q2.save_model('data/critics/{}#{}#SAC{}.pt'.format(params.env_name, params.team_name, str(j)))

    simulation.env.close()
    chrono.stop()


if __name__ == '__main__':
    args = get_args()
    print(args)
    create_data_folders()
    main(args)
    plot_results(args)
