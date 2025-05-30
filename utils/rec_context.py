class RecContextManager:
    def __init__(self, cold_start_episodes: int = 10):
        self.total_steps = 0
        self.cold_start  = cold_start_episodes

    def reset(self):
        self.total_steps = 0

    def use_fixed_quota(self) -> bool:
        return self.total_steps < self.cold_start

    def step(self):
        self.total_steps += 1


def get_recommendation_quota(
    user_pref: dict,
    context: RecContextManager,
    max_total: int = 6,
    min_per_type: int = 1
) -> dict:
    types = list(user_pref.keys())
    if context.use_fixed_quota():
        eq = max_total // len(types)  # 6//3 == 2
        return {t: eq for t in types}

    # 후기 모드: 선호 비율에 따라 할당
    raw = {t: round(user_pref[t] * max_total) for t in types}
    for t in types:
        raw[t] = max(raw[t], min_per_type)
    # 합이 초과하면 높은 초과도 순서로 차감
    while sum(raw.values()) > max_total:
        diff = {t: raw[t] - user_pref[t]*max_total for t in types}
        t_max = max(diff, key=diff.get)
        raw[t_max] -= 1
    # 합이 부족하면 가장 선호도 높은 타입에 추가
    while sum(raw.values()) < max_total:
        t_max = max(user_pref, key=user_pref.get)
        raw[t_max] += 1
    return raw