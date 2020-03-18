import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
tf.keras.backend.set_floatx('float64')
from misc import *
from collections import deque
from nets import Q1, Actor
import random


def real_training(run_id, lr = 10**-2,
 states_wasted=10**3, batch_size=64, epochs=10, buffer_size=10**3, predict_q_table=True):

    q1=Q1()
    q2=Q2()
    actor = Actor()

    optimizer_critic = tf.keras.optimizers.Adam(lr = lr)
    optimizer_actor = tf.keras.optimizers.Adam(lr = lr)

    pt=[]
    rt=[]
    rts = []
    loss_ev = []
    history = []
    history_would_have_done=[]

    betas_test = np.arange(-1,1,.01)
    optimal = max(ps(betas_test))
    buffer = ReplayBuffer(buffer_size=buffer_size)

    for episode in tqdm(range(states_wasted)):

        beta_would_do = np.squeeze(actor.give_action().numpy())
        pt.append(ps(beta_would_do))
        beta = beta_would_do + np.random.uniform(-.25, .25)
        history.append(beta)
        history_would_have_done.append(beta_would_do)
        reward = np.random.choice([1.,0.],1, p=[ps(beta), 1-ps(beta)])[0]
        rt.append(reward)
        rts.append(np.sum(rt))
        buffer.add(beta, reward)

        actions_did, rewards = buffer.sample(batch_size)

        if episode >0:
            with tf.GradientTape() as tape:
                tape.watch(q1.trainable_variables)
                predictions = q1(np.expand_dims(np.array(actions_did),axis=1))
                loss_sum = tf.keras.losses.MSE(np.expand_dims(np.array(rewards),axis=1),predictions)
                loss = tf.reduce_mean(loss_sum)
                grads = tape.gradient(loss, q1.trainable_variables)
                optimizer_critic.apply_gradients(zip(grads, q1.trainable_variables))
                loss_ev.append(np.squeeze(loss.numpy()))

            with tf.GradientTape() as tape:
                actions = actor(np.expand_dims(np.zeros(batch_size),axis=1))
                tape.watch(actions)
                qvals = q1(actions)
            dq_da = tape.gradient(qvals, actions)

            with tf.GradientTape() as tape:
                actions = actor(np.expand_dims(np.zeros(batch_size),axis=1))
                theta = actor.trainable_variables
            da_dtheta = tape.gradient(actions, theta, output_gradients=-dq_da)
            optimizer_actor.apply_gradients(zip(da_dtheta, actor.trainable_variables))

        else:
            loss_ev.append(0)
    rtsum = rts/np.arange(1,len(rts)+1)
    if predict_q_table:
        predictions = q1(np.expand_dims(np.array(betas_test),axis=1))
    else:
        predictions=None

    plot_evolution(rt=rtsum, pt=pt, optimal=optimal, betas=betas_test,
     preds=predictions , loss=loss_ev, history_betas=history, history_would_have_done=history_would_have_done,
      run_id= run_id)

    data = "buffer_size: {}\Batch size: {}\nLearning_rate: {}\nOptimizer Critic: {}\nOptimizer Actor: {}".format(str(buffer_size), str(batch_size), str(lr),optimizer_critic.__str__(),optimizer_actor.__str__()  )

    os.chdir(run_id)
    with open("info.txt", 'w') as f:
        f.write(data)
        f.close()
    os.chdir("..")
    return


check_folder("mate")

for k in range(1):
    run_id=record()
    number_run = "run_"+str(run_id)
    real_training(run_id=number_run, lr=1e-4,
     states_wasted=10**4, batch_size=500, buffer_size=2*10**3)
os.chdir("..")

# for lrs in [1e-3, 1e-2, 1e-4]:
#     check_folder("lrs"+str(batch_size))
#     for k in range(2):
#         run_id=record()
#         number_run = "run_"+str(run_id)
#         real_training(run_id=number_run, lr=lrs,
#          states_wasted=10**5, batch_size=500, buffer_size=10**3)
#     os.chdir("..")



####
