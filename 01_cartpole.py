"""
倒立摆
"""
import gym
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc

# 创建倒立摆环境
env = gym.make("CartPole-v0")
state = env.reset()  # 重置环境的状态为𝑆0
done = False  # 游戏是否结束，初始值为False，也就是未结束
episode_rewards = []  # 每回合奖励[𝑅0, 𝑅1, ⋯, 𝑅𝑇]
gamma = 0.95  # 折扣因子𝛾
total_reward = 0  # 带折扣因子的回报（总奖励）
frames = []  # 保存每一帧

# done=True时结束
while not done:
    # 渲染画面，并保存帧
    frames.append(env.render(mode="rgb_array"))

    # 随机选择一个动作𝐴𝑡，0：向左推，1：向右推
    action = random.choice([0, 1])

    # 下一个状态𝑆𝑡+1，即时奖励𝑅𝑡，是否结束，_
    next_state, reward, done, _ = env.step(action)

    # [𝑅0, 𝑅1, ⋯, 𝑅𝑇]
    episode_rewards.append(reward)

# 逆序计算𝐺𝑡 = 𝑅𝑡 + 𝛾𝐺𝑡+1
for r in episode_rewards[::-1]:
    # 𝐺𝑡 = 𝑅𝑡 + 𝛾𝐺𝑡+1
    total_reward = r + gamma * total_reward

print("total_reward:", total_reward)
env.close()


def show_animation(imgs):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    frames = []

    for i, img in enumerate(imgs):
        frame = [ax.imshow(img, animated=True)]
        frame.append(ax.text(10, 20, f"Step: {i + 1}", animated=True))  # Step数表示
        frames.append(frame)
    ax.axis("off")

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)

    # 保存动画
    ani.save(r"..\visualization\cartpole.gif",
             writer="pillow")

    plt.close(fig)
    return ani


show_animation(frames)
