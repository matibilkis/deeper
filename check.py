import numpy as np

rews = np.load("rews.npy")
times = np.arange(1,len(rews)+1)
# cum_rews = []
# for i in range(len(rews)):
#     cum_rews.append(np.sum(rews[:(i+1)])/(i+1))

# print(cum_rews/np.array(times))
# print(cum_rews)
import matplotlib.pyplot as plt
# print()
plt.plot(np.array(range(10000)),rews/np.array(range(10000)))
# plt.plot(times, cum_rews/np.array(times))
plt.show()
