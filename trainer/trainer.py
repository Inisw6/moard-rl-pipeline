import numpy as np

class Trainer:
    def __init__(self, env, meta_agent, content_agent, config,
                 state_builder, content_embedder):
        self.env = env
        self.meta = meta_agent
        self.ctn = content_agent
        self.cfg = config
        self.state_builder = state_builder
        self.content_embedder = content_embedder

    def train(self):
        episodes = self.cfg['train']['episodes']
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            total_r = 0
            while not done:
                # 1) 상위 액션
                ma = self.meta.select_action(state)

                # 2) 하위 상태 생성 - builder 사용
                c_state = self.state_builder.build(state, ma)

                # 3) 하위 액션
                ca = self.ctn.select_action(c_state)

                # 4) env step
                next_state, r, done, _ = self.env.step((ma, ca))
                total_r += r

                # 5) next_c_state 도 builder로 생성
                next_c_state = self.state_builder.build(next_state, ma)

                # 6) store & learn
                self.meta.store(state, ma, r, next_state, done)
                self.ctn.store(c_state, ca, r, next_c_state, done)

                self.meta.learn()
                self.ctn.learn()
                

                state = next_state

            print(f"Episode {ep} TotalReward {total_r:.2f}")

