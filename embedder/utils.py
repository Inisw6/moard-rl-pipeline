import numpy as np

def time_bucket_fn(dt):
    """입력: datetime → 아침/점심/저녁/밤 one-hot"""
    hour = dt.hour
    if 5 <= hour < 11: bucket = 0   # 아침
    elif 11 <= hour < 16: bucket = 1  # 점심
    elif 16 <= hour < 21: bucket = 2  # 저녁
    else: bucket = 3  # 밤
    onehot = [0, 0, 0, 0]
    onehot[bucket] = 1
    return np.array(onehot)