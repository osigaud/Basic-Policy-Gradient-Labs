import os

from arguments import get_args
from chrono import Chrono
from critics import DoubleQNet
from environment import make_env
from memory import ReplayBuffer
from policies.policy_net import PolicyNet
from simulation import Simulation
from visu.visu_critics import plot_critic
from visu.visu_policies import plot_policy
from visu.visu_results import plot_results
from wrappers.policy_wrapper import PolicyWrapper


def create_data_folders() -> None:
    """Create folders where to save output files if they are not already there."""
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
    """Create files to save the policy loss and the critic loss.

    Args:
        study_name (str): The name of the study.
        env_name (str): The name of the environment.
    """
    policy_loss_name = "data/save/policy_loss_" + study_name + '_' + env_name + ".txt"
    policy_loss_file = open(policy_loss_name, "w")
    critic_loss_name = "data/save/critic_loss_" + study_name + '_' + env_name + ".txt"
    critic_loss_file = open(critic_loss_name, "w")
    return policy_loss_file, critic_loss_file


def main(params) -> None:
    env = make_env(params.env_name, params.policy_type, params.max_episode_steps, params.env_obs_space_name)
    env.set_file_name("{}_{}".format(params.gradients[0], params.env_name))

    simulation = Simulation(env, params.nb_trajs, params.update_threshold, params.nb_updates, params.batch_size,
                            params.print_interval)
    simulation.rescale_reward = lambda reward: reward / 10

    policy_loss_file, critic_loss_file = set_files(params.gradients[0], params.env_name)

    chrono = Chrono()

    for j in range(params.nb_repet):
        env.reinit()
        memory = ReplayBuffer()

        # Initialise the policy/actor
        policy = PolicyNet(params.lr_actor, params.init_alpha, params.lr_alpha, params.target_entropy_alpha)
        pw = PolicyWrapper(policy, params.policy_type, params.env_name, params.team_name, params.max_episode_steps)
        pw.duration_flag = False
        # Initialise the critics
        critic = DoubleQNet(params.lr_critic,params.gamma, params.tau)

        plot_policy(policy, env, True, params.env_name, params.study_name, '_ante_', j, plot=False)

        simulation.train(memory, pw, critic, policy_loss_file, critic_loss_file)

        plot_policy(policy, env, True, params.env_name, params.study_name, '_post_', j, plot=False)
        plot_critic(env, params.env_name, critic.q1, policy, params.study_name, '_q1_post_', j)
        plot_critic(env, params.env_name, critic.q2, policy, params.study_name, '_q2_post_', j)
        critic.q1.save_model('data/critics/{}#{}#SAC{}.pt'.format(params.env_name, params.team_name, str(j)))
        critic.q2.save_model('data/critics/{}#{}#SAC{}.pt'.format(params.env_name, params.team_name, str(j)))

    simulation.env.close()
    chrono.stop()


if __name__ == '__main__':
    args = get_args()
    print(args)
    create_data_folders()
    main(args)
    plot_results(args)
