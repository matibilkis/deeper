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
 states_wasted=10**3, min_ep=1, batch_size=10, epochs=10, buffer_size=10**3):

    q1=Q1()
    optimizer = tf.keras.optimizers.Adam(lr = lr)
    pt=[]
    rt=[]
    rts = []

    betas = np.arange(-1,0,1/number_betas)
    ntable=np.zeros(len(betas))

    optimal = max(ps(betas))
    buffer = ReplayBuffer(buffer_size=buffer_size)
    for episode in tqdm(range(states_wasted)):
        ep = max(np.exp(-episode/100),min_ep)
        label, beta = greedy_action(q1, betas, ep)
        ntable[label]+=1
        reward = np.random.choice([1.,0.],1, p=[ps(beta), 1-ps(beta)])[0]
        rt.append(reward)
        rts.append(np.sum(rt))
        buffer.add(beta, reward)

        if (episode > batch_size):
            for k in range(epochs): #loop epoch
                actions_did, rewards = buffer.sample(batch_size)
                with tf.device("/cpu:0"):
                    with tf.GradientTape() as tape:
                        tape.watch(q1.trainable_variables)
                        predictions = q1(np.expand_dims(np.array(actions_did),axis=1))
                        loss_sum = tf.keras.losses.MSE(predictions,np.expand_dims(np.array(rewards),axis=1))
                        loss = tf.reduce_mean(loss_sum)
                        grads = tape.gradient(loss, q1.trainable_variables)
                        optimizer.apply_gradients(zip(grads, q1.trainable_variables))
                        pt.append(ps(greedy_action(q1,betas,ep=0)[1]))

        else:
            pt.append(0.5)


    rtsum = rts/np.arange(1,states_wasted+1)
    predictions = q1.prediction(betas)
    plot_evolution(rt=rtsum, pt=pt, optimal=optimal, betas=betas, preds=predictions , run_id= run_id)

    data = "betas: " +str(betas)+"\nBernoulli rewards!\nOptimizer: "+optimizer.__str__()+"\nBatch_size: "+str(batch_size)
    data = data + "\nLearning rate: "+str(lr)+"\nEpsilon decaying! e^{-t/100}" + "\nBuffer size: "+str(buffer.buffer_size)+"\nMin_ep: "+str(min_ep)+"\nEpochs: "+str(epochs)

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

real_training(run_id=number_run, lr=10**-2,
 states_wasted=10**4, min_ep=1, batch_size=5, epochs=1, buffer_size=5)














####
