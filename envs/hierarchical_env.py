import gym
from gym import spaces
import numpy as np

class HierarchicalRecEnv(gym.Env):
    def __init__(self, config, user_embedder, reward_function):
        super().__init__()
        self.embedder = user_embedder
        self.reward_function = reward_function
        self.max_steps = config['env']['max_steps']

        # very simple user_state: just zeros
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.embedder.output_dim(),)
        )
        # meta: 3종류, content: up to 10 후보
        self.meta_action_space = spaces.Discrete(config['meta_agent']['action_dim'])
        self.content_action_space = spaces.Discrete(config['content_agent']['action_dim'])
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        self.current_user = self._generate_fake_user()
        state = self.embedder.embed(self.current_user)
        return state

    def step(self, action_tuple):
        meta_a, content_a = action_tuple
        self.step_count += 1
        # reward: 임의로 랜덤
        content_info = self._get_fake_content(meta_a, content_a)
        reward = self.reward_function.calculate(content_info)

        done = self.step_count >= self.max_steps
        # next_state: still zeros for now
        next_state = self.embedder.embed(self._generate_fake_user())
        
        info = {}
        return next_state, reward, done, info

    def _generate_fake_user(self):
        from datetime import datetime
        return {
            "recent_logs": [
                {"category": ["금융"], "emotion": 0.8, "dwell": 25, "type": "YouTube"},
                {"category": ["테크"], "emotion": 0.6, "dwell": 15, "type": "Blog"},
            ],
            "current_time": datetime.now()
        }

    def _get_fake_content(self, meta_action, content_action):
        return {
            "clicked": np.random.choice([0, 1]),
            "emotion": np.random.uniform(-1, 1),  # 감정 점수
            "dwell": np.random.uniform(0, 30)     # 체류시간 (초)
        }