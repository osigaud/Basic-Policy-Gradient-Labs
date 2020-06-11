
def plot_trajectory(nbsteps, trajs, actor, env, figure=None, figure_file="trajectory.png", definition=50, plot=True, save_figure=False,):
    if figure is None:
        plt.figure(figsize=(10, 10))

    for i in range(len(trajs)):
        plt.scatter(trajs[i]["x"], trajs[i]["y"], c=range(1, len(trajs[i]["x"]) + 1), s=3)
    plt.colorbar(orientation="horizontal", label="steps")

    if env.observation_space.shape[0] != 2:
        raise(ValueError("Observation space of dimension {}, should be 2".format(env.observation_space.shape[0])))

    # Add the actor phase portrait
    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high
    # Use the dimension names if given otherwise default to "x" and "y"
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            portrait[definition - (1 + index_y), index_x] = actor.predict(np.array([[x, y]]))

    # TODO: Use the `corner` parameter
    plt.imshow(portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect='auto')
    plt.colorbar(label="action")
    plt.title("Actor and trajectory at step {}".format(nbsteps))
    # Add a point at the center
    plt.scatter([-0.5], [0])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(figure_file)
    plt.close()