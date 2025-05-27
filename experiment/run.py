import yaml
from envs.hierarchical_env import HierarchicalRecEnv
from agents.meta_agent import MetaAgent
from agents.content_agent import ContentAgent
from trainer.trainer import Trainer

from embedder.simple_embedder import SimpleUserEmbedder
from embedder.utils import time_bucket_fn

from agents.dqn_meta_agent import DQNMetaAgent

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))

    # UserEmbedder 정의
    category_list = ["금융", "테크", "정치", "연예", "라이프"]
    user_embedder = SimpleUserEmbedder(category_list, time_bucket_fn)

    # state_dim 자동 반영
    user_state_dim = user_embedder.output_dim()
    meta_action_dim = cfg["meta_agent"]["action_dim"]
    content_action_dim = cfg["content_agent"]["action_dim"]

    content_state_dim = user_state_dim + meta_action_dim

    # 3. env + agent 초기화
    env = HierarchicalRecEnv(cfg, user_embedder)
    # meta = MetaAgent(action_dim=meta_action_dim, state_dim=user_state_dim)
    meta = DQNMetaAgent(
        state_dim=user_embedder.output_dim(),
        action_dim=cfg["meta_agent"]["action_dim"],
        cfg=cfg
    )
    ctn = ContentAgent(action_dim=content_action_dim, state_dim=content_state_dim)

    trainer = Trainer(env, meta, ctn, cfg)
    trainer.train()

if __name__ == "__main__":
    main()
