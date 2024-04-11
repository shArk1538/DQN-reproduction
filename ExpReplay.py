import random
from collections import namedtuple, deque

Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)  # deque实现的效果与注释代码相同
        
        # self.capacity = capacity
        # self.memory = []
        # self.position = 0   

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

        # if len(self.memory) < self.capacity:
        #     self.memory.append(None)
        # self.memory[self.position] = Transition(*args)
        # self.position = (self.position + 1) % self.capacity # 新的替换掉最久远的

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)# 从经验池中随机采样

    def __len__(self):
        return len(self.memory)
    

if __name__ == "__main__":
    de = deque([1,2,3], maxlen=4)
    de.append(4)
    print(de)
    de.append(5)
    print(de)