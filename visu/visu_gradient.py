
import matplotlib.pyplot as plt
import imageio


def visu_replay_data(list_states, list_targets) -> None:
    '''
    visualize, for a list of states plotted for their first dimension, 
    the corresponding target value for the critic as computed either with
    a TD or a MC method.
    in the case of the MC method, it gives the value V(s) of being in that state
    in the TD case, the traget is given by the local temporal difference error
    the state is assumed 4-dimensional (cartpole case)
    :param list_states: a list of states, usually taken from a batch
    :param list_targets: a list of target values, usually computed from a batch
    :return: a picture of the data
    '''
    bx1, bx2, bx3, bx4 = zip(*list_states)
    plt.figure(figsize=(10, 4))
    plt.scatter(bx1, list_targets, color="blue")
    plt.title('Regression Analysis')
    plt.xlabel('Feature')
    plt.ylabel('Value and target')
    plt.savefig('./regress/data_regress.pdf')
    plt.show()


def visu_loss_along_time(cpts, losses, loss_file_name) -> None:
    '''
    Plots the evolution of the loss along time
    :param cpts: step counter
    :param losses: the successive values of the loss
    :return: plots an image
    '''
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.cla()
    # ax.set_ylim(-1.0, 500.0)  # data dependent
    ax.set_title('Loss Analysis', fontsize=35)
    ax.set_xlabel('cpt', fontsize=24)
    ax.set_ylabel('loss', fontsize=24)
    ax.scatter(cpts, losses, color="blue", alpha=0.2)
    plt.savefig('./results/' + loss_file_name + '.pdf')
    plt.show()


def make_gif_replay_data(my_images, features, targets, vals) -> None:
    '''
    Plots in an animated gif the (feature, target) and (feature, current value) pairs
    Makes it possible to see how approximation improves along time
    :param my_images: a list of images. Must pass [] the first time
    :param features: the values of on state features in the dataset
    :param targets:  the values of target in the dataset
    :param vals:  the current values in the dataset
    :return: an animated gif
    '''
    plt.figure(figsize=(10, 4))
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.cla()
    ax.set_title('Regression Analysis', fontsize=35)
    ax.set_xlabel('State', fontsize=24)
    ax.set_ylabel('Value', fontsize=24)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-10.1, 500.2)
    ax.scatter(features, targets.data.numpy(), color="blue", alpha=0.2)
    ax.scatter(features, vals.data.numpy(), color='green', alpha=0.5)
    ax.text(0.0, 90.0, 'Epoch = %d' % epoch,
            fontdict={'size': 24, 'color': 'red'})
    ax.text(-2.0, 90.0, 'Loss = %.4f' % mean_loss,
            fontdict={'size': 24, 'color': 'red'})

    # Used to return the plot as an image array
    # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    my_images.append(image)
    imageio.mimsave('./regress/regress_V.gif', my_images, fps=12)
