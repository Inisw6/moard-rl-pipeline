import gym
from gym import spaces
import numpy as np

class HierarchicalRecEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.max_steps = config['env']['max_steps']
        # very simple user_state: just zeros
        self.observation_space = spaces.Box(-1, 1, shape=(config['meta_agent']['state_dim'],), dtype=np.float32)
        # meta: 3종류, content: up to 10 후보
        self.meta_action_space = spaces.Discrete(config['meta_agent']['action_dim'])
        self.content_action_space = spaces.Discrete(config['content_agent']['action_dim'])
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        # dummy user state
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action_tuple):
        meta_a, content_a = action_tuple
        self.step_count += 1
        # reward: 임의로 랜덤
        reward = np.random.randn()
        done = self.step_count >= self.max_steps
        # next_state: still zeros for now
        next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return next_state, reward, done, info
