import torch
import math
import numpy as np
from itertools import count
import matplotlib.pyplot as plt


EPS_START = 1   # ε-greedy的起始值
EPS_END = 0.02  # ε-greedy的终止值
EPS_DECAY = 1000000  # 控制ε的指数衰减率(越高越慢)
INITIAL_MEMORY = 10000
TARGET_UPDATE = 1000


MODEL_STORE_PATH = './models'
modelname = 'DQN_Breakout'

# RENDER = False  # 不渲染游戏画面
RENDER = True  # 渲染游戏画面

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, env, agent, n_episode):
        self.env = env
        self.n_episode = n_episode
        self.agent = agent
        # self.losslist = []
        self.rewardlist = []
        self.avg_rewardlist = []


    # 获取当前状态，将env返回的状态通过transpose调换轴后作为状态
    def get_state(self,obs):
        # print(obs.shape)
        state = np.array(obs)
        # state = state.transpose((1, 2, 0)) #将2轴放在0轴之前
        state = torch.from_numpy(state)
        return state.unsqueeze(0)    # 转化为四维的数据结构


    # 训练智能体
    def train(self):
        for episode in range(self.n_episode):

            obs = self.env.reset()  # 环境初始化
            # print('============obs = self.env.reset()============')
            # state = self.img_process(obs)
            state = np.stack((obs[0], obs[1], obs[2], obs[3]))
            # print(state.shape)
            state = self.get_state(state)
            # print(state.shape)
            episode_reward = 0.0

            # print('episode:',episode)
            for t in count():
                # print(state.shape)
                action = self.agent.select_action(state)  # epsilon-greedy选择动作

                if RENDER:
                    self.env.render()  # 渲染游戏画面

                obs,reward,done,_,_ = self.env.step(action)
                episode_reward += reward

                if not done:
                    # next_state = self.get_state(obs)

                    # next_state = self.img_process(obs)
                    next_state = np.stack((obs[0], obs[1], obs[2], obs[3]))
                    next_state = self.get_state(next_state)
                else:
                    next_state = None
                # print(next_state.shape)
                reward = torch.tensor([reward], device=device)

                # 将四元组存到memory中
                '''
                state: batch_size channel h w    size=batch_size * 4
                action: size=batch_size * 1
                next_state: batch_size channel h w    size=batch_size * 4
                reward: size=batch_size * 1
                '''
                self.agent.memory_buffer.push(state, action.to('cpu'), next_state, reward.to('cpu')) # 里面的数据都是Tensor

                # 进入下一状态
                state = next_state
                
                # 经验池满了之后开始学习
                if self.agent.stepdone > INITIAL_MEMORY:
                    self.agent.learn()
                    if self.agent.stepdone % TARGET_UPDATE == 0:
                        print('======== target DQN updated =========')
                        # [hard-update]1000步后将Q网络中的参数θ硬拷贝到target_net中 (更新延后)
                        self.agent.target_DQN.load_state_dict(self.agent.DQN.state_dict())

                if done:
                    break

            agent_epsilon = EPS_END + (EPS_START - EPS_END)* math.exp(-1. * self.agent.stepdone / EPS_DECAY)

            print('Total steps: {} \t Episode/steps: {}/{} \t Total reward: {} \t Avg reward: {} \t epsilon: {}'.format(
                self.agent.stepdone, episode, t, episode_reward, episode_reward/t, agent_epsilon))


            if episode % 1000 == 0:  # 每1000幕保存.pt模型
                torch.save(self.agent.DQN.state_dict(), MODEL_STORE_PATH + '/' + "{}_episode{}.pt".format(modelname, episode))
                # print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.agent.stepdone, episode, t, episode_reward))

            self.rewardlist.append(episode_reward)
            self.avg_rewardlist.append(episode_reward/t)

            self.env.close()
        return
    
    #绘制单幕总奖励曲线
    def plot_total_reward(self):
        plt.plot(self.rewardlist)
        plt.xlabel("Training epochs")
        plt.ylabel("Total reward per episode")
        plt.title('Total reward curve of DQN on Skiing')
        plt.savefig('DQN_train_total_reward.png')
        plt.show()

    #绘制单幕平均奖励曲线
    def plot_avg_reward(self):
        plt.plot(self.avg_rewardlist)
        plt.xlabel("Training epochs")
        plt.ylabel("Average reward per episode")
        plt.title('Average reward curve of DQN on Skiing')
        plt.savefig('DQN_train_avg_reward.png')
        plt.show()