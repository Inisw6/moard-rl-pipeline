import torch
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseAgent
from models.q_network import QNetwork
from replay.replay_buffer import ReplayBuffer

class DQNContentAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, cfg):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-networks
        self.q_net        = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # optimizer & buffer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg["train"]["lr"])
        self.buffer    = ReplayBuffer(capacity=cfg["train"].get("buffer_size", 10000))

        # hyperparams
        tcfg             = cfg["train"]
        self.gamma       = tcfg.get("gamma", 0.99)
        self.batch_size  = tcfg.get("batch_size", 64)
        self.epsilon     = tcfg.get("epsilon_start", 1.0)
        self.epsilon_min = tcfg.get("epsilon_min", 0.05)
        self.epsilon_dec = tcfg.get("epsilon_decay", 0.995)
        self.update_freq = tcfg.get("update_freq", 100)
        self.step_count  = 0

    def select_action(self, state):
        self.step_count += 1
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qv = self.q_net(st)
        return qv.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        # to tensor
        S  = torch.FloatTensor(states).to(self.device)
        A  = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        R  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        S2 = torch.FloatTensor(next_states).to(self.device)
        D  = torch.FloatTensor(dones.astype(np.float32)).unsqueeze(1).to(self.device)

        Q  = self.q_net(S).gather(1, A)
        with torch.no_grad():
            Q2 = self.target_q_net(S2).max(1)[0].unsqueeze(1)
            target = R + self.gamma * Q2 * (1 - D)

        loss = F.mse_loss(Q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε decay
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)
        # target network sync
        if self.step_count % self.update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
