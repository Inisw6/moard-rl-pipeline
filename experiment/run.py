import yaml
from envs.hierarchical_env import HierarchicalRecEnv
from agents.meta_agent import MetaAgent
from agents.content_agent import ContentAgent
from trainer.trainer import Trainer

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    env = HierarchicalRecEnv(cfg)
    meta = MetaAgent(cfg)
    ctn = ContentAgent(cfg)
    trainer = Trainer(env, meta, ctn, cfg)
    trainer.train()

if __name__ == "__main__":
    main()
