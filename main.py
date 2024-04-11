import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from Preprocessing import atari_preprocess
from DQN_Agent import Agent
from Train import Trainer


# ============= hyperparameters =============
BATCH_SIZE = 32  # replay buffer中采样transitions的数量
GAMMA = 0.99  # Reward的折扣因子
EPS_START = 1   # ε-greedy的起始值
EPS_END = 0.1  # ε-greedy的终止值
EPS_DECAY = 1000000  # 控制ε的指数衰减率(越高越慢)
EPS_RANDOM_COUNT = 50000 # 前50000步纯随机用于探索
TARGET_UPDATE = 1000 # target network的更新步长
lr = 2.5e-4 # 2.5e-4 in paper
INITIAL_MEMORY = 10000
MEMORY_SIZE = 10 * INITIAL_MEMORY
n_episode = 100000 # 100000
# ===========================================


env = gym.make("ALE/Breakout-v5", render_mode = "human")  # 创建游戏环境 
env = atari_preprocess(env)

action_space = env.action_space
print(action_space)

MODEL_PATH = './models/DQN_Breakout_episode9000.pt'  # 可以加载已经训练好的模型，为空则直接训练
agent = Agent(in_channels=4, action_space=action_space, learning_rate=lr, memory_size=MEMORY_SIZE,trained_model_path=MODEL_PATH)

trainer = Trainer(env, agent, n_episode)
trainer.train()  # 经验池满后开始采样训练


#========================== 性能评估 ==========================
trainer.plot_total_reward()
trainer.plot_avg_reward()

# save total reward list
np.save('total_reward_list_breakout_1e5.npy',np.array(trainer.rewardlist))
# np.save('avg_reward_list_breakout_1e5.npy',np.array(trainer.avg_rewardlist))

print('The training costs {} episodes'.format(len(trainer.rewardlist)))

print('The max episode reward is {}, at episode {}'.format(
    max(trainer.rewardlist),
    trainer.rewardlist.index(max(trainer.rewardlist))
    ))


# 合并 5 episodes为 1 episode
assert(len(trainer.rewardlist)%5==0)
reshaped_reward_array = np.array(trainer.rewardlist).reshape((int(len(trainer.rewardlist)/5), 5))
# 沿着第二个维度求和
summed_rewawrd_array = reshaped_reward_array.sum(axis=1)

print('Now takes 5 episodes as 1, the training cost {} complete episodes'.format(len(summed_rewawrd_array)))

print('The max episode return is {}, at episode {}'.format(
    max(summed_rewawrd_array),
    np.where(summed_rewawrd_array == max(summed_rewawrd_array))
    ))


# 合并 200 episodes为 1 episode
assert(len(summed_rewawrd_array)%200==0)
reshaped_reward_array_200 = summed_rewawrd_array.reshape((int(len(summed_rewawrd_array)/200), 200))
# 沿着第二个维度求和
summed_rewawrd_array_200 = reshaped_reward_array_200.sum(axis=1)
avg_rewawrd_array_200 = summed_rewawrd_array_200/200.0

np.save('avg_rewawrd_array_200_breakout_1e5.npy',avg_rewawrd_array_200)

print('The following graph takes 1000 games as 1 epoch where 5 games equals to 1 episode as stated before')

max_idx = np.argmax(avg_rewawrd_array_200)
max_y = max(avg_rewawrd_array_200)
print('The best average return per epoch is {}, at epoch {}'.format(max_idx,max_y))

plt.figure(figsize=(10,6))
plt.plot(avg_rewawrd_array_200,marker='o',markersize=4)
plt.xlabel("Training Epochs",fontsize=12)
plt.ylabel("Average Reward per Episode",fontsize=12)

plt.scatter(max_idx, max_y, color='red', s=60)
plt.annotate(f'max avg return: ({max_idx}, {max_y:.2f})', xy=(max_idx, max_y), xytext=(max_idx-40, max_y-1),
             arrowprops=dict(facecolor='red', shrink=0.05))
# plt.title('Average Reward of DQN on Breakout')
plt.savefig('DQN_train_total_reward_Breakout.svg')
plt.show()