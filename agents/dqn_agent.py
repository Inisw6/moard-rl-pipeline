import random, numpy as np, torch
from models.q_network import QNetwork
from replay.replay_buffer import ReplayBuffer
from agents.base_agent import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self, user_dim, content_dim, config, device='cpu'):
        super().__init__()
        self.device      = torch.device("cuda" if torch.cuda.is_available() else device)
        # Q-network: state–action pair input
        self.q_net        = QNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net = QNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config["train"]["lr"])
        self.buffer    = ReplayBuffer(config["replay"]["capacity"])

        self.gamma       = config["agent"]["gamma"]
        self.batch_size  = config["train"]["batch_size"]
        self.epsilon     = config["agent"]["eps_start"]
        self.epsilon_min = config["agent"]["eps_min"]
        self.epsilon_dec = config["agent"]["eps_decay"]
        self.update_freq = config["agent"].get("update_freq", 100)
        self.step_count  = 0

    def select_action(self, user_state, candidate_embs):
        """
        user_state: np.ndarray( user_dim )
        candidate_embs: List[np.ndarray(content_dim)]
        """
        self.step_count += 1
        if random.random() < self.epsilon:
            return random.randrange(len(candidate_embs))

        us = torch.FloatTensor(user_state).unsqueeze(0).to(self.device)  # [1, user_dim]
        # batch 처리: [K, user_dim], [K, content_dim]
        us_expanded = us.repeat(len(candidate_embs), 1)                   # [K, user_dim]
        ce = torch.FloatTensor(candidate_embs).to(self.device)           # [K, content_dim]
        with torch.no_grad():
            q_vals = self.q_net(us_expanded, ce).squeeze(1).cpu().numpy() # [K]
        return int(np.argmax(q_vals))

    def store(self, user_state, content_emb, reward, next_state, next_cands_embs, done):
        # next_cands_embs: dict{type->List[emb]}
        self.buffer.push((user_state, content_emb),
                         reward,
                         (next_state, next_cands_embs),
                         done)

    def learn(self):
        # 버퍼에 충분한 샘플이 있어야 학습 시작
        if len(self.buffer) < self.batch_size:
            return

        # sample()이 반환하는 순서대로 unpack 합니다:
        # user_states, cont_embs, rewards, next_info, dones
        user_states, cont_embs, rewards, next_info, dones = self.buffer.sample(self.batch_size)
        # next_info는 (next_states, next_cands_embs_dicts)
        next_states, next_cands_embs_dicts = next_info

        # Tensor로 변환
        us = torch.FloatTensor(user_states).to(self.device)            # [B, user_dim]
        ce = torch.FloatTensor(cont_embs).to(self.device)              # [B, content_dim]
        rs = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)   # [B, 1]
        ds = torch.FloatTensor(dones).unsqueeze(1).to(self.device)     # [B, 1]

        # 1) Q(s,a)
        q_sa = self.q_net(us, ce)                                      # [B, 1]

        # 2) target = r + γ * max_a' Q'(s', a')
        max_next_q_list = []
        for ns, next_cands in zip(next_states, next_cands_embs_dicts):
            # ns: next user state (np.ndarray)
            # next_cands: dict of type->List[content_emb]
            # flatten all type별 후보 임베딩
            all_embs = sum(next_cands.values(), [])

            # 각 후보마다 Q' 계산
            us_next   = torch.FloatTensor(ns).unsqueeze(0).to(self.device)
            usn_rep   = us_next.repeat(len(all_embs), 1)               # [K', user_dim]
            ce_next   = torch.FloatTensor(all_embs).to(self.device)    # [K', content_dim]
            with torch.no_grad():
                qn = self.target_q_net(usn_rep, ce_next).squeeze(1)    # [K']
            max_next_q_list.append(qn.max())                          # 스칼라 텐서

        # [B] -> [B,1]
        max_next_q = torch.stack(max_next_q_list).unsqueeze(1)        # [B,1]

        # 3) Bellman 타겟 계산
        target = rs + self.gamma * max_next_q * (1 - ds)              # [B,1]

        # 4) MSE loss & 백워드
        loss = torch.nn.functional.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε decay & 타겟 네트워크 sync
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)
        if self.step_count % self.update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        st = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(st)
        self.target_q_net.load_state_dict(st)
