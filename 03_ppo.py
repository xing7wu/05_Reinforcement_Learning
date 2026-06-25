"""
近端策略优化
"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    """策略神经网络"""

    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class ValueNet(nn.Module):
    """价值神经网络"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lam = 0.95
        self.epsilon = 0.2
        self.lr_pi = 0.001
        self.lr_v = 0.02
        self.action_size = 2
        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        # 动作的概率分布
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        # 创建一个二项分布采样器
        m = Categorical(probs)
        # 采样动作𝜋𝜃(𝑎𝑡|𝑠𝑡)
        action = m.sample().item()

        return action, probs

    def collect_trajectory(self, env):
        """采样一条轨迹"""
        state = env.reset()
        states, next_states, actions, action_probs, rewards, dones = [], [], [], [], [], []
        done = False

        while not done:
            action, probs = self.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)  # S_t
            next_states.append(next_state)  # S_(t+1)
            actions.append(action)  # A_t
            action_probs.append(probs[action])  # π_old(a_t|s_t)
            rewards.append(reward)  # R_t
            dones.append(done)  # done_t

            state = next_state

        # states: [S_0, S_1, S_2, ..., S_(T-1)]
        # next_states: [S_1, S_2, S_3, ..., S_T]
        # actions: [A_0, A_1, A_2, ..., A_(T-1)]
        # action_probs：[π_old(a_0|s_0),π_old(a_1|s_1)...]
        # rewards: [R_0, R_1, R_2, ..., R_(T-1)]
        # dones: [False, False, False, ..., True]
        return states, next_states, actions, action_probs, rewards, dones

    def update(self, trajectory):
        """用轨迹trajectory数据更新策略网络和价值网络"""
        states, next_states, actions, action_probs, rewards, dones = trajectory

        states = torch.tensor(states)
        next_states = torch.tensor(next_states)
        actions = torch.tensor(actions).view(-1, 1)
        rewards = torch.tensor(rewards).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1)

        # [𝑉 (𝑠0), 𝑉(𝑠1), …, 𝑉(𝑠𝑇−1)]
        v = self.v(states).detach()

        # [TD-target0, TD - target1, …, TD - target𝑇−1]
        td_target = rewards + self.gamma * self.v(next_states) * (1 - dones)  # TD-target𝑡 = 𝑅𝑡 + 𝛾𝑉 (𝑠𝑡+1)

        # [𝛿0, 𝛿1, …, 𝛿𝑇−1]
        td_delta = td_target - v  # 𝛿𝑡 = 𝑅𝑡 + 𝛾𝑉 (𝑠𝑡+1) − 𝑉 (𝑠𝑡)

        # 计算每个时刻t的广义优势估计（GAE）：[𝐴GAE0, 𝐴GAE1, …, 𝐴GAE𝑇−1]
        gae = self.compute_gae(td_delta.cpu())

        # 冻结一份旧策略采取动作的对数概率：log π_old(a_t|s_t)
        old_probs = torch.tensor(action_probs).view(-1, 1)
        old_log_probs = torch.log(old_probs).detach()

        for _ in range(10):
            # 新策略采取动作的对数概率: log π_θ(a_t|s_t)
            log_probs = torch.log(self.pi(states).gather(1, actions))

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * gae
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * gae

            # 𝜋loss
            loss_pi = torch.mean(-torch.min(surr1, surr2))
            # 𝑉loss
            loss_v = F.mse_loss(self.v(states), gae + v)

            self.optimizer_pi.zero_grad()
            self.optimizer_v.zero_grad()
            loss_v.backward()
            loss_pi.backward()
            self.optimizer_pi.step()
            self.optimizer_v.step()

    def compute_gae(self, td_delta):
        # δ_t
        td_delta = td_delta.detach().numpy()

        gae_list = []
        last_gae = 0.0
        # GAE_t = δ_t + γλGAE_(t+1)
        for delta in td_delta[::-1]:
            last_gae = delta + self.gamma * self.lam * last_gae
            gae_list.append(last_gae)
        gae_list.reverse()
        return torch.tensor(gae_list)


env = gym.make("CartPole-v0")
env.seed(23)
torch.manual_seed(23)
agent = Agent()
return_list = []
episode_list = []

for episode in range(500):
    trajectory = agent.collect_trajectory(env)
    agent.update(trajectory)

    return_list.append(sum(trajectory[4]))
    episode_list.append(episode)
    if (episode + 1) % 10 == 0:
        print(f"回合：{episode + 1}, 总奖励：{sum(trajectory[4])}")


def plot_reward(episode_list, return_list, filename):
    """绘制奖励图像"""
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    plt.show()
    f.savefig(filename, bbox_inches="tight")


plot_reward(episode_list, return_list, r"..\visualization\ppo_reward.pdf")
