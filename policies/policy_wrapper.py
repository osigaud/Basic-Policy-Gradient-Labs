import os


class PolicyWrapper:
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
