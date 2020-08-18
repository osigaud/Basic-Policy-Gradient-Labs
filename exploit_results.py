import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from arguments import make_full_string, get_args
import seaborn as sns
sns.set()



# old stuff
def plot_durations(durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_data(filename, label):
    """
    generic plot function to return a curve from a file with an index and a number per line
    importantly, several datasets can be stored into the same file
    and the curve will contain a variance information based on the repetition
    :param filename: the file containing the data
    :param label: the label to be shown in the plot (a string)
    :return: a curve with some variance and a label, embedded in plt. 
    """
    data = pd.read_csv(filename, sep=' ')
    data = pd.read_csv(filename, sep=' ', names=list(range(data.shape[1])))
    x1 = list(data.groupby([0]).quantile(0.75)[1])
    x2 = list(data.groupby([0]).quantile(0.25)[1])
    x_mean = list(data.groupby([0]).mean()[1])
    x_std = list(data.groupby([0]).std()[1])
    plt.plot(x_mean, label=label)
    plt.fill_between(list(range(len(x1))), x1, x2, alpha=0.25)
    return x_mean, x_std


def plot_from_file(filename, label) -> None:
    """
    generic plot function to return a curve from a file with just one number per line
    :param filename: the file containing the data
    :param label: the label to be shown in the plot (a string)
    :return: a curve with a label, embedded in plt.
    """
    with open(filename, 'r') as file:
        data = [float(x) for x in file]
    # print(data)
    plt.plot(data, label=label)


# to be refreshed
def exploit_beta(params):
    path = os.getcwd() + "/data/save"
    for beta in [0.1, 0.5, 1.0, 5.0, 10.0]:
        name = "/progress_" + str(beta) + '.txt'
        plot_data(path + name, str(beta))

    plot_data(path + "/progress.txt", "normalized discounted rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
    plt.savefig(path + '/../results/' + make_full_string(params) + '.pdf')
    plt.show()


# specific to AC study
def exploit_duration_ac():
    path = os.getcwd() + "/data/save"
    plot_data(path + "/progress_ac.txt", "durations")

    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
    plt.savefig(path + '/../results/durations_ac.pdf')
    plt.show()


# generic
def exploit_duration_regress(params):
    path = os.getcwd() + "/data/save"
    plot_data(path + "/duration_regress_" + params.env_name + '.txt', "durations")

    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
    plt.savefig(path + '/../results/durations_regress_' + make_full_string(params) + '.pdf')
    plt.show()


def exploit_reward_regress(params):
    path = os.getcwd() + "/data/save"
    plot_data(path + "/reward_regress_" + params.env_name + '.txt', "reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
    plt.title(params.env_name)
    plt.savefig(path + '/../results/rewards_regress_' + make_full_string(params) + '.pdf')
    plt.show()


def exploit_duration_full(params):
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/duration_" + study[i] + '_' + params.env_name + '.txt', "duration " + study[i])

    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
    plt.title(params.env_name)
    plt.savefig(path + '/../results/durations_' + make_full_string(params) + 'pg.pdf')
    plt.show()


def exploit_critic_loss_full(params):
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/critic_loss_" + study[i] + '_' + params.env_name + '.txt', "critic loss " + study[i])

    plt.xlabel("Cycles")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")  # , bbox_to_anchor=(1, 0.5)
    plt.title(params.env_name)
    plt.savefig(path + '/../results/critic_loss_' + make_full_string(params) + 'pg.pdf')
    plt.show()


def exploit_policy_loss_full(params):
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/policy_loss_" + study[i] + '_' + params.env_name + '.txt', "policy loss " + study[i])

    plt.xlabel("Cycles")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
    plt.title(params.env_name)
    plt.savefig(path + '/../results/policy_loss_' + make_full_string(params) + 'pg.pdf')
    plt.show()


def exploit_nstep(params):
    path = os.getcwd() + "/data/save"
    steps = [1, 5, 10, 15, 20]

    for j in ['loss', 'reward', 'duration']:
        mean_list = []
        std_list = []
        for i in steps:
            mean, std = plot_data(path + '/' + j + '_nstep_' + str(i) + '_'
                                  + params.env_name + '.txt', j + '_nstep_' + str(i))
            # print('n:', i, ' mean :', mean[-1], ' std:', std[-1])
            mean_list.append(mean[-1])
            std_list.append(std[-1])

        plt.title(params.env_name)
        plt.xlabel("Episodes")
        plt.ylabel(j)
        plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
        plt.savefig(path + '/../results/' + j + '_nstep_' + make_full_string(params) + '.pdf')
        plt.show()

        plt.plot(steps, mean_list, label="bias")
        plt.plot(steps, std_list, label="variance")
        plt.title(params.env_name)
        plt.xlabel("N in N-step")
        plt.ylabel('variance, bias')
        plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
        plt.savefig(path + '/../results/bias_variance_' + j + '_' + make_full_string(params) + '.pdf')
        plt.show()


def check_nstep(params):
    path = os.getcwd() + "/data/save"
    study1 = 'batchTD'
    for j in ['loss', 'reward', 'duration']:
        for i in [1]:
            mean, std = plot_data(path + "/" + j + '_nstep_' + str(i) + '_' + params.env_name + '.txt', j + '_nstep_' + str(i))
            print('n:', i, ' mean :', mean[-1], ' std:', std[-1])
        plot_data(path + "/" + j + '_' + study1 + '_' + params.env_name + '.txt', 'loss ' + study1)

        plt.title(params.env_name)
        plt.xlabel("Episodes")
        plt.ylabel(j)
        plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
        plt.savefig(path + '/../results/' + j + '_nstep_check.pdf')
        plt.show()


def exploit_nstep_diff(params):
    path = os.getcwd() + "/data/save"
    steps = [1, 5, 10, 20]
    mean_list = []
    std_list = []
    for i in steps:
        mean, std = plot_data(path + '/diff_' + str(i) + '_' + params.env_name + '.txt', 'nstep_' + str(i))
        # print('n:', i, ' mean :', mean[-1], ' std:', std[-1])
        mean_list.append(mean[-1])
        std_list.append(std[-1])

    plt.title(params.env_name)
    plt.xlabel("Episodes")
    plt.ylabel("diff")
    plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
    plt.savefig(path + '/../results/diff_nstep_' + make_full_string(params) + '.pdf')
    plt.show()

    plt.plot(steps, mean_list, label="bias")
    plt.plot(steps, std_list, label="variance")
    plt.title(params.env_name)
    plt.xlabel("N in N-step")
    plt.ylabel('variance, bias')
    plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
    plt.savefig(path + '/../results/bias_variance_' + make_full_string(params) + '.pdf')
    plt.show()


def custom_plot(params):
    path = os.getcwd() + "/data/save"
    study = ['batchTD', 'batchMC', 'nstep']
    valid = ['', '_valid']
    content = ['loss', 'targ']

    for c in content:
        for s in study:
            for v in valid:

                plot_data(path + '/' + c + s + v + '_' + params.env_name + '.txt', c + '_' + s + v)

        plt.title(params.env_name)
        plt.ylabel(c)
        plt.xlabel("Episodes")

        plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
        plt.savefig(path + '/../results/' + c + '_' + make_full_string(params) + '.pdf')
        plt.show()


def exploit_reward_full(params):
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/reward_" + study[i] + '_' + params.env_name + '.txt', "reward " + study[i])

    plt.title(params.env_name)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
    plt.savefig(path + '/../results/rewards_' + make_full_string(params) + '.pdf')
    plt.show()


def main_exploit(params):
    if params.study_name == "pg":
        exploit_duration_full(params)
        exploit_reward_full(params)
        exploit_policy_loss_full(params)
        exploit_critic_loss_full(params)
    elif params.study_name == "regress":
        exploit_duration_regress(params)
        exploit_reward_regress(params)
    elif params.study_name == "ac":
        exploit_duration_ac()
    elif params.study_name == "loss":
        exploit_critic_loss_full(params)
        exploit_policy_loss_full(params)
    elif params.study_name == "nstep":
        exploit_nstep(params)
    elif params.study_name == "diff":
        exploit_nstep_diff(params)
    elif params.study_name == "target":
        custom_plot(params)
    else:
        print("exploit unknown case :", params.study_name)

if __name__ == '__main__':
    args = get_args()
    print(args)
    # args.gradients = ["baseline"]
    # os.chdir("./save")
    # args.env_name = "Pendulum-v0"
    main_exploit(args)
    # args.env_name = "CartPole-v0"
    # exploit_critic_loss_full(args)
    # exploit_reward_full(args)
    # custom_plot('CartPoleContinuous-v0')
    # check_nstep("CartPoleContinuous-v0")
    # exploit_nstep_diff("CartPoleContinuous-v0")
    # check_nstep("CartPole-v0")
    # exploit_nstep("CartPoleContinuous-v0")
    # main_exploit("CartPoleContinuous-v0", 'pg', ['q_baseline'])
    # main_exploit("CartPoleContinuous-v0",'loss', ['batchTD', 'batchMC','batchTD_valid', 'batchMC_valid'])
