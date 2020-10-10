import numpy as np
import tensorflow as tf
from tqdm import tqdm

#data_all, labels_all = np.load("ProfilesSet/data.npy"),  np.load("ProfilesSet/label.npy")
#dat_set = tf.data.Dataset.from_tensor_slices((data_all,labels_all))
#data_all_test=tf.convert_to_tensor(data_all)
#labels_all_test = tf.convert_to_tensor(labels_all)
#dat_set.shuffle(buffer_size=10**4)
data_all=list(range(1600))

pp=[]
for bs in [8,64, 32,512]:

    cuts = len(data_all)/bs
    tot = 10**5/cuts
    p=0
    for kk in range(int(tot)):
        p+=cuts
    pp.append(p)

print(pp)
