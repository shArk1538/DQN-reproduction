import ExpReplay
from ExpReplay import ReplayMemory
from Network import DQN

import torch
import torch.optim as optim
import torch.nn.functional as F

import math, random

EPS_START = 1   # ε-greedy的起始值
EPS_END = 0.02  # ε-greedy的终止值
EPS_DECAY = 1000000  # 控制ε的指数衰减率(越高越慢)
EPS_RANDOM_COUNT = 50000 # 前50000步纯随机用于探索
BATCH_SIZE = 32
GAMMA = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, in_channels=4, action_space=[], learning_rate=1e-4, memory_size=10000, epsilon=1, trained_model_path=''):
        
        self.in_channels = in_channels
        self.action_space = action_space
        self.action_dim = self.action_space.n

        self.memory_buffer = ReplayMemory(memory_size)

        self.stepdone = 0

        self.DQN = DQN(self.in_channels, self.action_dim).to(device)          # policy network
        self.target_DQN = DQN(self.in_channels, self.action_dim).to(device)   # target network
        
        # 可加载已训练好的模型
        if(trained_model_path != ''):
            self.DQN.load_state_dict(torch.load(trained_model_path))

        self.target_DQN.load_state_dict(self.DQN.state_dict())
        self.optimizer = optim.RMSprop(self.DQN.parameters(),lr=learning_rate, eps=0.001, alpha=0.95)
    

    def select_action(self, state):

        self.stepdone += 1
        state = state.to(device)
        epsilon = EPS_END + (EPS_START - EPS_END)* \
            math.exp(-1. * self.stepdone / EPS_DECAY)   # 随机选择动作系数epsilon衰减
        
        # epsilon-greedy策略选择动作
        if self.stepdone<EPS_RANDOM_COUNT or random.random()<epsilon:
            action = torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
        else:
            action = self.DQN(state).detach().max(1)[1].view(1,1)  # 选择Q值最大的动作并view

        return action
    

    def learn(self):
        '''
        optimize the model
        '''
        if self.memory_buffer.__len__()<BATCH_SIZE:
            return # 经验池小于BATCH_SIZE则直接返回

        transitions = self.memory_buffer.sample(BATCH_SIZE)  # 从经验池中采样
        '''
        batch.state - tuple of all the states (each state is a tensor)           (BATCH_SIZE * channel * h * w)
        batch.next_state - tuple of all the next states (each state is a tensor) (BATCH_SIZE * channel * h * w)
        batch.reward - tuple of all the rewards (each reward is a float)         (BATCH_SIZE * 1)
        batch.action - tuple of all the actions (each action is an int)          (BATCH_SIZE * 1)
        '''

        batch = ExpReplay.Transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))


        # 判断是不是在最后一个状态，最后一个状态的next设置为None
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8).bool()

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(device)

        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        
        # 计算当前状态的Q值
        state_action_values = self.DQN(state_batch).gather(1, action_batch)  # Q(s,a)

        # 计算{t+1}状态V值
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        # .detach(): 从计算图中分离，不进行backpropagate
        next_state_values[non_final_mask] = self.target_DQN(non_final_next_states).max(1)[0].detach()
        
        # 计算期望的Q值
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # 计算Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()  # 反向传播

        # 梯度剪切，限制权值更新幅度
        for param in self.DQN.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

if __name__ == '__main__':
    pass