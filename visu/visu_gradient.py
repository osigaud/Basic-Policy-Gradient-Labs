import matplotlib.pyplot as plt


def visu_replay_data(list_states, list_targets) -> None:
    """
    visualize, for a list of states plotted for their first dimension, 
    the corresponding target value for the critic as computed either with
    a TD or a MC method.
    In the MC case, it gives the value V(s) of being in that state
    In the TD case, the target is given by the local temporal difference error
    the state is assumed 4-dimensional (cartpole case)
    :param list_states: a list of states, usually taken from a batch
    :param list_targets: a list of target values, usually computed from a batch
    :return: nothing
    """
    bx1, bx2, bx3, bx4 = zip(*list_states)
    plt.figure(figsize=(10, 4))
    plt.scatter(bx1, list_targets, color="blue")
    plt.title('Regression Analysis')
    plt.xlabel('Feature')
    plt.ylabel('Value and target')
    plt.savefig('./regress/data_regress.pdf')
    plt.show()


def visu_loss_along_time(cpts, losses, loss_file_name) -> None:
    """
    Plots the evolution of the loss along time
    :param cpts: step counter
    :param losses: the successive values of the loss
    :param loss_file_name: the file where to store the results
    :return: nothing
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.cla()
    # ax.set_ylim(-1.0, 500.0)  # data dependent
    ax.set_title('Loss Analysis', fontsize=35)
    ax.set_xlabel('cpt', fontsize=24)
    ax.set_ylabel('loss', fontsize=24)
    ax.scatter(cpts, losses, color="blue", alpha=0.2)
    plt.savefig('./results/' + loss_file_name + '.pdf')
    plt.show()