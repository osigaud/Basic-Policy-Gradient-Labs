import os


class PolicyWrapper:
    """
    This class is used to perform evaluation of a policy without any assumption on the nature of the policy. 
    It contains the information about the training environment and the team name
    which are necessary to display the result of evaluations. 
    These two informations are stored into the file name when saving the policy to be evaluated.
    """
    def __init__(self, policy, policy_type, env_name, team_name, max_steps):
        self.policy = policy
        self.policy_type = policy_type
        self.env_name = env_name
        self.team_name = team_name
        self.max_steps = max_steps

    def save(self, score=0) -> None:
        """
        Save the model into a file whose name contains useful information for later evaluation
        :param score: the score of the network
        :return: nothing
        """
        directory = os.getcwd() + '/data/policies/'
        filename = directory + self.env_name + '#' + self.team_name + '_' + str(score) \
                   + '#' + self.policy_type + '#' + str(self.max_steps)+ '#' + str(score) + '.pt'
        self.policy.save_model(filename)

    def load(self, filename):
        """
        Load a model from a file whose name contains useful information for evaluation (environment name and team name)
        :param filename: the file name, including the path
        :return: the obtained network
        """
        fields = filename.split('#')
        tmp = fields[0]
        env_name = tmp.split('/')
        self.env_name = env_name[-1]
        self.team_name = fields[1]
        self.policy_type = fields[2]
        #### MODIF : check if max steps is None
        if fields[3] != "None":
            self.max_steps = int(fields[3])
        else:
            self.max_steps = None
        ####
        net = self.policy.load_model(filename)
        return net
