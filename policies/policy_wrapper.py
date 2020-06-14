import os


class PolicyWrapper:
    """
    This class is used to perform evaluation of a policy without any assumption on the nature of the policy. 
    It contains the information about the training environment and the team name
    which are necessary to display the result of evaluations. 
    These two informations are stored into the file name when saving the policy to be evaluated.
    """
    def __init__(self, policy, name, env_name):
        self.policy = policy
        self.team_name = name
        self.env_name = env_name

    def save(self, score=0):
        directory = os.getcwd() + '/data/policies/'
        filename = directory + self.env_name + '#' + self.team_name + '_' + str(score) + '#' + str(score) + '.pt'
        self.policy.save_model(filename)

    def load(self, filename):
        fields = filename.split('#')
        tmp = fields[0]
        envname = tmp.split('/')
        self.env_name = envname[-1]
        self.team_name = fields[1]
        net = self.policy.load_model(filename)
        return net
