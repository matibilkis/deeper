import os
import tensorflow as tf
import numpy as np
from nets import RNN

from tqdm import tqdm
import matplotlib.pyplot as plt

name="task2_dset"
data, labels = np.load(name+"/data.npy"), np.load(name+"/label.npy")

stoppings = []
for k in range(4):
    for kk in range(10**k,10**(k+1),10**k):
        stoppings.append(kk)
for j in range(1,4):
    stoppings.append(j*10**4)

d = tf.data.Dataset.from_tensor_slices((data, labels))
d.shuffle(buffer_size=10**4)


def fun(bs):
    l=[]
    cuts = len(data)/bs
    ind=0
    dons=[]

    net = RNN(seed_val=0.5)
    if not os.path.exists("grads_{}".format(bs)):
        os.makedirs("grads_{}".format(bs))
    l=[]
    logdir1 = "logs/scalars/training/{}".format(bs)
    logdir2 = "logs/scalars/testing/{}".format(bs)

    train_loss = tf.summary.create_file_writer(logdir1)
    test_loss = tf.summary.create_file_writer(logdir2)

    st=0
    for k in tqdm(range(int(3*10**4/cuts))):
        for k1,k2 in list(d.batch(bs)):
            st+=1
            if st in stoppings:
                lo, gr = net.train_step(k1, k2, return_gradients=True)
                np.save("grads_{}/{}".format(bs,k),gr)
                net.save_weights("net_{}/".format(bs)+str(k))
                dons.append(k)
                l.append(lo)
            else:
                lo =net.train_step(k1, k2)
                l.append(lo)
            with train_loss.as_default():
                tf.summary.scalar('train_loss', l[-1], step=st)
            with test_loss.as_default():
                tf.summary.scalar('test_loss',tf.reduce_mean(tf.keras.losses.MeanSquaredError()(net(data),tf.expand_dims(labels, axis=-1))), step=st)

    os.system("python3 image_gen.py --names {}".format(bs))
    means=[]
    maxs=[]
    for ep in dons:
        gr = np.load("grads_{}/{}.npy".format(bs,ep), allow_pickle=True)
        means.append([np.mean(gr[ko].numpy()) for ko in range(len(gr))])
        maxs.append([np.max(np.abs(gr[ko].numpy())) for ko in range(len(gr))])


    ti=[]
    for kkk in np.log10(dons):
        if kkk in list(range(10)):
            ti.append(kkk)

    plt.figure(figsize=(10,30))
    axs={}
    for ik in range(9):
        axs[str(ik)] = plt.subplot2grid((9,2),(ik,0))
        axs[str(ik)].plot(np.log10(dons),np.array(means)[:,ik], label="mean gradient at layer {}".format(ik), color="red",alpha=0.75, linewidth=4)
        axs[str(ik)].set_xticks(ti)
        axs[str(ik)].set_xticklabels(["10^{}".format(int(kik)) for kik in ti])
        if k==0:
            axs[str(ik)].set_title("mean value of gradient \nat each layer", size=20)
        axs[str(ik)] = plt.subplot2grid((9,2),(ik,1))
        axs[str(ik)].plot(np.log10(dons),np.array(maxs)[:,ik], label="max absolute value of gradient at layer {}".format(ik), color="red",alpha=0.75, linewidth=4)
        axs[str(ik)].set_xticks(ti)
        axs[str(ik)].yaxis.tick_right()
        axs[str(ik)].set_xticklabels(["10^{}".format(int(kik)) for kik in ti])
        if k==0:
            axs[str(ik)].set_title("max abs value of gradient \nat each layer", size=20)
    plt.savefig("evolution_bs_{}.png".format(bs))

for kk in [8,64, 512, 1600, 3200]:
    fun(kk)
# import multiprocessing as mp
# with mp.Pool(4) as p:
#     p.map(fun, [len(data), len(data), len(data)])
#
