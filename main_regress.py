from chrono import Chrono
from simu import make_simu_from_params
from policies import BernoulliPolicy, NormalPolicy, SquashedGaussianPolicy, PolicyWrapper
from critics import VNetwork, QNetworkContinuous
from arguments import get_args
from visu.visu_policies import plot_policy
from visu.visu_critics import plot_critic
from visu.visu_results import plot_results
from main_pg import create_data_folders, set_files
from mountain_car_expert import regress


def study_regress(params) -> None:
    assert params.policy_type in ['bernoulli', 'normal', 'squashedGaussian'], 'unsupported policy type'
    chrono = Chrono()
    study = params.gradients
    simu = make_simu_from_params(params)
    for i in range(len(study)):
        simu.env.set_file_name(study[i] + '_' + simu.name)
        policy_loss_file, critic_loss_file = set_files(study[i], simu.name)
        print("study : ", study[i])
        for j in range(params.nb_repet):
            simu.env.reinit()
            if params.policy_type == "bernoulli":
                policy = BernoulliPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            elif params.policy_type == "normal":
                policy = NormalPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            elif params.policy_type == "squashedGaussian":
                policy = SquashedGaussianPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            pw = PolicyWrapper(policy, params.team_name, simu.name)
            plot_policy(policy, simu.env, True, simu.name, study[i], '_ante_', j, plot=False)

            if not simu.discrete:
                act_size = simu.env.action_space.shape[0]
                critic = QNetworkContinuous(simu.obs_size + act_size, 24, 36, 1, params.lr_critic)
            else:
                critic = VNetwork(simu.obs_size, 24, 36, 1, params.lr_critic)
            # plot_critic(simu, critic, policy, study[i], '_ante_', j)

            regress(simu, policy, params.policy_type, params.render)
            simu.train(pw, params, policy, critic, policy_loss_file, critic_loss_file, study[i])
            plot_policy(policy, simu.env, True, simu.name, study[i], '_post_', j, plot=False)
            plot_critic(simu, critic, policy, study[i], '_post_', j)
            critic.save_model('data/critics/' + params.env_name + '#' + params.team_name + '#' + study[i] + str(j) + '.pt')
    chrono.stop()


def main():
    args = get_args()
    print(args)
    create_data_folders()
    args.gradients = ['sum']
    study_regress(args)
    plot_results(args)

if __name__ == '__main__':
    main()
