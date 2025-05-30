import numpy as np

class SimpleConcatBuilder:
    def __init__(self, user_dim: int = 30, content_dim: int = 5):
        self.user_dim    = user_dim
        self.content_dim = content_dim

    def output_dim(self) -> int:
        return self.user_dim

    def embed_user(self, user: dict) -> np.ndarray:
        """
        user: {"recent_logs": [...], "current_time": ...}
        반환: [emotion_avg, dwell_avg, yt_ratio, blog_ratio, news_ratio, padding...]
        """
        logs = user["recent_logs"]
        emotion_avg = np.mean([l["emotion"] for l in logs])
        dwell_avg   = np.mean([l["dwell"] for l in logs])

        # 소문자 타입 카운트
        counts = {"youtube": 0, "blog": 0, "news": 0}
        for l in logs:
            counts[l["type"].lower()] += 1
        total = len(logs)
        type_vec = np.array([
            counts["youtube"] / total,
            counts["blog"]    / total,
            counts["news"]    / total
        ])

        # 패딩 포함 최종 벡터
        vec = np.concatenate([
            [emotion_avg, dwell_avg],
            type_vec,
            np.zeros(self.user_dim - 5)
        ])
        return vec.astype(np.float32)

    def embed_content(self, content: dict) -> np.ndarray:
        """
        content: {'id': ..., 'type': 'youtube'|'blog'|'news', 'emotion': float, 'dwell': float}
        반환: [emotion, dwell_norm, onehot_youtube, onehot_blog, onehot_news]
        """
        emotion = content.get("emotion", 0.0)
        dwell   = content.get("dwell", 0.0) / 30.0  # 예시 정규화
        type_onehot = {
            "youtube": [1, 0, 0],
            "blog":    [0, 1, 0],
            "news":    [0, 0, 1]
        }[content.get("type", "youtube")]

        vec = np.array([emotion, dwell] + type_onehot)
        if len(vec) < self.content_dim:
            vec = np.concatenate([vec, np.zeros(self.content_dim - len(vec))])
        return vec.astype(np.float32)

    def estimate_preference(self, user_state: np.ndarray) -> dict:
        """
        user_state[2:5] 에서 콘텐츠 비율을 읽어 선호도로 사용
        """
        yt, bl, nw = user_state[2], user_state[3], user_state[4]
        return {
            "youtube": float(yt),
            "blog":    float(bl),
            "news":    float(nw),
        }