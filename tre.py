import pickle
import numpy as np

d={}
for k in range(20):
    d[str([k])] = np.random.randn(430,10)

a_file = open("data.pkl", "wb")

pickle.dump(d, a_file)
a_file.close()

opp = open("data.pkl", "rb")
output = pickle.load(opp)

print(output)
