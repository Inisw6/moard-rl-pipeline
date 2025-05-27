import numpy as np

class Trainer:
    def __init__(self, env, meta_agent, content_agent, config):
        self.env = env
        self.meta = meta_agent
        self.ctn = content_agent
        self.cfg = config

    def train(self):
        episodes = self.cfg['train']['episodes']
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            total_r = 0
            while not done:
                # 1) 상위 액션
                ma = self.meta.select_action(state)
                # 2) 하위 상태/인풋: (state + one-hot meta) 간단 스텁
                meta_onehot = np.eye(self.meta.action_dim)[ma]
                c_state = np.concatenate([state, meta_onehot])
                # 3) 하위 액션
                ca = self.ctn.select_action(c_state)
                # 4) env step
                next_state, r, done, _ = self.env.step((ma, ca))
                total_r += r
                # 5) store & learn
                self.meta.store(state, ma, r, next_state, done)
                next_c_state = np.concatenate([next_state, meta_onehot])
                self.ctn.store(c_state, ca, r, next_c_state, done)
                # self.ctn.store(c_state, ca, r, next_state, done)
                self.meta.learn()
                self.ctn.learn()
                state = next_state
            print(f"Episode {ep} TotalReward {total_r:.2f}")
