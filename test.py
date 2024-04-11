import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


env = gym.make("BreakoutNoFrameskip-v4") # 原始无修改游戏环境，210×160
env = AtariPreprocessing(env) # 84×84，frame-skip=4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(n_observations, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7*7*64, 512)
        self.fc5 = nn.Linear(512, n_actions)
    
    def forward(self, input):
        x = input.float() / 255  # 将uint8[0,255]范围的像素归一化到[0,1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # fc4为全连接层，将输出展平
        x = F.relu(self.fc4(x))
        output = self.fc5(x)

        return output

BATCH_SIZE = 32  # replay buffer中采样transitions的数量
GAMMA = 0.99  # Reward的折扣因子
EPS_START = 1  # ε-greedy的起始值
EPS_END = 0.1  # 终止值
EPS_DECAY = 1000  # 控制ε的指数衰减率(越高越慢)
TAU = 0.005  # target network的更新率
LR = 2.5e-4  # RMSProp的学习率

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.RMSprop(policy_net.parameters(), lr=LR, momentum=0.95, eps=0.001, alpha=0.95)  # 使用RMSProp
memory = ReplayMemory(10000)  # 经验池capacity=10000

# 动作选择函数
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)  # 随机选择动作系数epsilon 衰减，也可以使用固定的epsilon
    steps_done += 1
    # ε-greedy策略选择动作
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = policy_net(state).max(1).indices.view(1, 1)  # 选择Q值最大的动作并view
    else:
        action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
    return action

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch






if __name__ == "__main__":
    env = gym.make("BreakoutNoFrameskip-v4")  # 创建原始游戏环境

    # print(env.unwrapped.get_action_meanings()) # 游戏动作的含义
    print(env.action_space.n) # 动作空间[0:NOOP 1:FIRE 2:RIGHT 3:LEFT]
    print(env.observation_space) # 原始210×160的RGB图像，每个像素点RGB范围[0,255]

    # env = AtariPreprocessing(env,frame_skip=4,grayscale_newaxis=True, scale_obs=True) # 实现论文中的预处理
    # print(env.observation_space) # 84×84

    observation, info = env.reset()
    print(info) # lives:游戏中剩余的生命数，frame_number:游戏进行了多少帧
    plt.imshow(observation) # 以图片的形式显示numpy.ndarray
    plt.show()
    