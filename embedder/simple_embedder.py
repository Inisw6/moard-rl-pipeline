import numpy as np
from collections import defaultdict
from embedder.base_user_embedder import BaseUserEmbedder

class SimpleUserEmbedder(BaseUserEmbedder):
    def __init__(self, category_list, time_bucket_fn):
        self.category_list = category_list
        self.time_bucket_fn = time_bucket_fn

    def embed(self, user_dict):
        logs = user_dict['recent_logs']
        now = user_dict['current_time']

        # 1. 카테고리 분포
        counter = defaultdict(int)
        for log in logs:
            for cat in log.get('category', []):
                counter[cat] += 1
        total = sum(counter.values())
        cat_vec = np.array([
            counter[cat] / total if total > 0 else 0.0
            for cat in self.category_list
        ])

        # 2. 감정 평균
        emotions = [log.get('emotion', 0.0) for log in logs]
        avg_emotion = np.mean(emotions) if emotions else 0.0

        # 3. 세션 길이
        session_len = len(logs)

        # 4. 시간대 one-hot
        time_vec = self.time_bucket_fn(now)

        return np.concatenate([
            cat_vec,
            [avg_emotion],
            [session_len],
            time_vec
        ])

    def output_dim(self):
        return len(self.category_list) + 1 + 1 + 4  # 카테고리 + 감정 + 길이 + 시간대
