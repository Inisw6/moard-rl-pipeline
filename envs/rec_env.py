import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.rec_context import RecContextManager, get_recommendation_quota

class RecEnv(gym.Env):
    def __init__(self, config, embedder, reward_fn):
        super().__init__()
        # 의존성 주입
        self.embedder  = embedder
        self.reward_fn = reward_fn
        self.context   = RecContextManager(config['env']['cold_start'])

        # 환경 설정
        self.max_steps  = config['env']['max_steps']
        self.top_k      = config['env']['top_k']
        self.step_count = 0

        # 상태 공간: 사용자 임베딩 차원
        dim = embedder.output_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.context.reset()

        user  = self._generate_fake_user()
        state = self.embedder.embed_user(user)
        return state, {}

    def get_candidates(self, user_state):
        """
        Trainer 쪽에서 호출: 
        'youtube', 'blog', 'news' 키로 모두 소문자 반환
        """
        out = {}
        for ctype in ['youtube', 'blog', 'news']:
            # 더미 풀 생성 (실제론 DB나 벡터 검색)
            pool = [
                {
                    'id': i,
                    'type': ctype,
                    'emotion': np.random.uniform(-1, 1),
                    'dwell':   np.random.uniform(0, 30)
                }
                for i in range(100)
            ]
            scores = np.random.rand(len(pool))
            idxs   = np.argsort(scores)[::-1][: self.top_k]
            cands  = [pool[i] for i in idxs]
            embs   = [self.embedder.embed_content(c) for c in cands]
            out[ctype] = (cands, embs)
        return out

    def step(self, action_tuple):
        """
        action_tuple: (content_type: str, idx: int)
        """
        content_type, idx = action_tuple
        self.step_count += 1

        # 현재 후보군에서 선택
        cands, _    = self.get_candidates(None)[content_type]
        selected    = cands[idx]
        reward      = self.reward_fn.calculate(selected)

        # 다음 상태 생성
        user_next  = self._generate_fake_user()
        next_state = self.embedder.embed_user(user_next)

        # 에피소드 종료 여부
        done = (self.step_count >= self.max_steps)
        self.context.step()

        return next_state, reward, done, {}

    def _generate_fake_user(self):
        from datetime import datetime
        return {
            "recent_logs": [
                {"category": ["금융"], "emotion": 0.8, "dwell": 25, "type": "YouTube"},
                {"category": ["테크"], "emotion": 0.6, "dwell": 15, "type": "Blog"},
            ],
            "current_time": datetime.now()
        }