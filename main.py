import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
tf.keras.backend.set_floatx('float64')
from misc import *
from collections import deque
from nets import Q1
import random


def real_training(run_id, number_betas=10, lr = 10**-2,
 states_wasted=10**3, min_ep=1, splits=10, epochs=10, buffer_size=10**3):

    q1=Q1()
    optimizer = tf.keras.optimizers.Adam(lr = lr)
    pt=[]
    rt=[]
    rts = []
    loss_ev = []
    history = []
    betas = np.arange(-1,0,1/number_betas)
    ntable=np.zeros(len(betas))

    optimal = max(ps(betas))
    buffer = ReplayBuffer(buffer_size=buffer_size)
    for episode in tqdm(range(states_wasted)):
        ep = max(np.exp(-episode/100),min_ep)
        label, beta = greedy_action(q1, betas, ep)
        history.append(beta)
        ntable[label]+=1
        reward = np.random.choice([1.,0.],1, p=[ps(beta), 1-ps(beta)])[0]
        rt.append(reward)
        rts.append(np.sum(rt))
        buffer.add(beta, reward)

        actions_did, rewards = buffer.sample(int(10*buffer_size/splits))
        # if episode%10==0:

        with tf.device("/cpu:0"):
            with tf.GradientTape() as tape:
                tape.watch(q1.trainable_variables)
                predictions = q1(np.expand_dims(np.array(actions_did),axis=1))
                loss_sum = tf.keras.losses.MSE(predictions,np.expand_dims(np.array(rewards),axis=1))
                loss = tf.reduce_mean(loss_sum)
                loss_ev.append(np.squeeze(loss.numpy()))
                grads = tape.gradient(loss, q1.trainable_variables)
                optimizer.apply_gradients(zip(grads, q1.trainable_variables))
                pt.append(ps(greedy_action(q1,betas,ep=0)[1]))
        # else:
        #     pt.append(0)
        #     loss_ev.append(0)

    rtsum = rts/np.arange(1,states_wasted+1)
    predictions = q1.prediction(betas)
    plot_evolution(rt=rtsum, pt=pt, optimal=optimal, betas=betas, preds=predictions , loss=loss_ev, history_betas=history, run_id= run_id)
    data = "buffer_size: {}\nSplits: {}\nNumber of betas: {}\nLearning_rate: {}\nOptimizer: {}".format(str(buffer_size), str(splits), str(len(betas)) + "- all: "+ str(betas), str(lr),optimizer.__str__()  )

    os.chdir(run_id)
    with open("info.txt", 'w') as f:
        f.write(data)
        f.close()

    os.chdir("/../..")
    return

if not os.path.exists("results"):
    os.makedirs("results")
os.chdir("results")

run_id=record()
number_run = "run_"+str(run_id)


# run_id, number_betas=10, lr = 10**-2,
#  states_wasted=10**4, min_ep=1, batch_size=10, epochs=10):

real_training(run_id=number_run, lr=0.01, number_betas=10,
 states_wasted=10**4, min_ep=1, splits=100, buffer_size=10**6)



####
