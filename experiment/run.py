import yaml
from envs.hierarchical_env import HierarchicalRecEnv
from agents.dqn_meta_agent import DQNMetaAgent
from agents.dqn_content_agent import DQNContentAgent
from trainer.trainer import Trainer

# Embedder
from embedder.simple_embedder import SimpleUserEmbedder
from embedder.utils import time_bucket_fn

# Reward Function
from reward.click_reward import ClickReward

# New: State Builder & Content Embedder
from state_builder.simple_concat_builder import SimpleConcatBuilder
from embedder.dummy_content_embedder import DummyContentEmbedder

def main():
    # 1. config load
    cfg = yaml.safe_load(open("config/config.yaml"))

    # 2. 전략 주입: 유저 임베더, 보상함수, 상태빌더, 콘텐츠임베더
    category_list = ["금융", "테크", "정치", "연예", "라이프"]
    user_embedder = SimpleUserEmbedder(category_list, time_bucket_fn)
    reward_fn = ClickReward()
    meta_action_dim = cfg["meta_agent"]["action_dim"]
    content_action_dim = cfg["content_agent"]["action_dim"]

    # 인터페이스 기반 전략 클래스 (확장성 중심)
    state_builder = SimpleConcatBuilder(meta_action_dim)
    content_embedder = DummyContentEmbedder(dim=10)

    # 3. env & agent init
    env = HierarchicalRecEnv(cfg, user_embedder, reward_fn)

    user_state = env.reset()
    print("[CHECK] user_state.shape =", user_state.shape)

    meta_state_dim = user_embedder.output_dim()
    content_state_dim = meta_state_dim + meta_action_dim  # builder가 반환할 state dim과 일치

    print("[CHECK] meta_state_dim =", meta_state_dim)
    print("[CHECK] content_state_dim =", content_state_dim)

    meta = DQNMetaAgent(
        state_dim=meta_state_dim,
        action_dim=meta_action_dim,
        cfg=cfg
    )

    ctn = DQNContentAgent(
        state_dim=content_state_dim,
        action_dim=content_action_dim,
        cfg=cfg
    )

    # 4. Trainer에 주입
    trainer = Trainer(env, meta, ctn, cfg,
                      state_builder=state_builder,
                      content_embedder=content_embedder)
    trainer.train()

if __name__ == "__main__":
    main()
