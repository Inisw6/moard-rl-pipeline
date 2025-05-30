import yaml
from envs.rec_env import RecEnv
from agents.dqn_agent import DQNAgent
from utils.rec_context import get_recommendation_quota

class Trainer:
    def __init__(self, config_path='config/config.yaml'):
        cfg = yaml.safe_load(open(config_path))

        # embedder, reward_fn 인스턴스화
        from embedder.simple_concat_builder import SimpleConcatBuilder
        from reward.reward_function import DefaultRewardFunction

        self.embedder  = SimpleConcatBuilder()
        self.reward_fn = DefaultRewardFunction()
        self.env       = RecEnv(cfg, self.embedder, self.reward_fn)
        self.agent     = DQNAgent(
            user_dim    = self.embedder.user_dim,
            content_dim = self.embedder.content_dim,
            config      = cfg
        )

    def run(self, num_episodes=1000):
        for ep in range(num_episodes):
            # 1) 에피소드 시작
            state, _ = self.env.reset()
            done     = False

            while not done:
                # 2) 현재 state 기반 후보군 & 임베딩 가져오기
                cand_dict = self.env.get_candidates(state)
                # cand_dict: {'youtube': (cands, cembs), 'blog': (...), 'news': (...)}

                # 3) 사용자 선호 추정 → quota 결정
                user_pref = self.embedder.estimate_preference(state)
                quota     = get_recommendation_quota(
                    user_pref,
                    self.env.context,
                    max_total=6
                )

                # 4) 유형별 추천 루프
                for ctype, cnt in quota.items():
                    cands, cembs = cand_dict[ctype]

                    for _ in range(cnt):
                        # 4-1) DQN으로 인덱스 선택
                        idx = self.agent.select_action(state, cembs)

                        # 4-2) Env.step 호출 (info는 이제 비어 있음)
                        next_state, reward, done, _ = self.env.step((ctype, idx))

                        # 4-3) next_state 기반으로 **후보군을 다시 불러와서** embeddings 준비
                        next_cand_dict = self.env.get_candidates(next_state)
                        # next_cands_embs 형태: {'youtube': [...], 'blog': [...], 'news': [...]}
                        next_cands_embs = {
                            t: emb for t, (_, emb) in next_cand_dict.items()
                        }

                        # 4-4) 경험 저장 & 학습
                        self.agent.store(
                            state,
                            cembs[idx],
                            reward,
                            next_state,
                            next_cands_embs,
                            done
                        )
                        self.agent.learn()

                        # 4-5) 상태 업데이트
                        state = next_state
                        if done:
                            break
                    if done:
                        break

            print(f"[Episode {ep+1:03d}] eps={self.agent.epsilon:.3f}")

if __name__ == "__main__":
    Trainer().run(num_episodes=500)