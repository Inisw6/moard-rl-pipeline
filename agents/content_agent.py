import numpy as np
from agents.base_agent import BaseAgent

class ContentAgent(BaseAgent):
    def __init__(self, config):
        self.action_dim = config['content_agent']['action_dim']

    def select_action(self, state):
        return np.random.randint(self.action_dim)

    def store(self, *args):
        pass

    def learn(self):
        pass
