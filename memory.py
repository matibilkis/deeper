import random
import numpy as np

class Memory():
    def __init__(self, max_memory, load_path=None):
        self._max_memory = max_memory
        self.samples = []
        if load_path != None:
            self.load_buffer(load_path)

    def add_sample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self._max_memory:
            self.samples.pop(0)
    def sample(self, no_samples):
        if no_samples > len(self.samples):
            return random.sample(self.samples, len(self.samples))
        else:
            return random.sample(self.samples, no_samples)


    def export(self, name="buffer"):
        np.save(name, np.array(self.samples), allow_pickle=True)
        print("memory saved under name: ", name)
        return

    def load_buffer(self,name="buffer.npy"):
        buffer = np.load(name, allow_pickle=True)
        for i in range(len(buffer)):
            self.add_sample(tuple(buffer[i]))
        return

    @property
    def num_samples(self):
        return len(self.samples)
