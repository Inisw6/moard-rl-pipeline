import numpy as np
from agents.base_agent import BaseAgent

class MetaAgent(BaseAgent):
    def __init__(self, config):
        self.action_dim = config['meta_agent']['action_dim']

    def select_action(self, state):
        # 완전 랜덤 예제
        return np.random.randint(self.action_dim)

    def store(self, *args):
        pass

    def learn(self):
        pass
