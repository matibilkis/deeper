import numpy as np
from collections import deque
import random


class ReplayBuffer():
    def __init__(self, buffer_size=10**3):
        self.buffer_size = buffer_size
        self.count = 0
        # self.betas = np.arange(-1.5,1.5,0.01)
        self.buffer = deque()
        # self.create_test_datasets()

    # def create_test_datasets(self):
    #     dt_l0 = []
    #     dt_l1 = []
    #
    #     for k in self.betas:
    #         dt_l0.append([k, ps_maxlik(k)])
    #         for n in [0.,1.]:
    #             for g in [-1.,1.]:
    #                 dt_l1.append([k, n, g, qval(k,n,g)])
    #     self.test_l0 = np.array(dt_l0)
    #     self.test_l1 = np.array(dt_l1)
    #     return

    def add(self, experience):
        #in particular is important you get (\”eta1, 01, \beta2, o2..., reward, guess)
        if not isinstance(experience, tuple):
            raise ValueError("buffer wants tuples!")
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, int(batch_size))
        # beta_batch, outcome_batch, guess_batch, r_batch= list(map(np.array, list(zip(*batch))))
        return np.array(list(map(np.array, list(zip(*batch))))).transpose()

    def clear(self):
        self.buffer.clear()
        self.count = 0
