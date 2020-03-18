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


import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
tf.keras.backend.set_floatx('float64')
from misc import *
from collections import deque
from nets import Q1, Actor, Q2
import random


def real_training(run_id, lr = 10**-2,
 states_wasted=10**3, batch_size=64, epochs=10, buffer_size=10**3, predict_q_table=True):

    q1=Q1()
    q2=Q2()
    actor = Actor()

    optimizer_critic = tf.keras.optimizers.Adam(lr = lr)
    optimizer_actor = tf.keras.optimizers.Adam(lr = lr)
    optimizer_l2 = tf.keras.optimizers.Adam(lr = lr)

    pt=[]
    rt=[]
    rts = []
    loss_ev = []
    history_betas = []
    history_would_have_done=[]

    betas_test = np.arange(-1,0,.05)
    optimal = max(ps_maxlik(betas_test))
    buffer = ReplayBufferKennedy(buffer_size=buffer_size)

    for episode in tqdm(range(states_wasted)):

        phase = np.random.choice([-1,1])

        beta_would_do = np.squeeze(actor.give_action().numpy())
        g=[]
        for n1 in [0,1]:
            g.append([-1,1][np.argmax(np.squeeze(q2( np.expand_dims(np.array([[beta_would_do, n1]]) ,axis=1 )).numpy()))])

        beta = beta_would_do + np.random.uniform(-.5,.5)
        history_betas.append(beta)
        history_would_have_done.append(beta_would_do)

        outcome = give_outcome(phase*0.4, beta)
        preds_guess = np.squeeze(q2( np.expand_dims(np.array([[beta, outcome]]) ,axis=1 )).numpy())
        guess = [-1,1][np.argmax(preds_guess)]

        pt.append(ps(beta_would_do,g))
        if guess == phase:
            reward=1
        else:
            reward = 0

        rt.append(reward)
        rts.append(np.sum(rt))
        buffer.add(beta, outcome, guess, reward)

        actions_did, outcomes, guessed, rewarded = buffer.sample(batch_size)

        histories = np.array([[be, out] for be, out in zip(actions_did, outcomes)])

        if episode >0:

            with tf.GradientTape() as tape:
                tape.watch(q2.trainable_variables)
                predictions_l2 = q2(np.expand_dims( histories, axis=1))
                loss=tf.keras.losses.binary_crossentropy(np.expand_dims(rewarded,axis=1), predictions_l2)
                # loss = tf.keras.losses.MSE(np.expand_dims(rewarded,axis=1), predictions_l2)
                loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, q2.trainable_variables)
                optimizer_l2.apply_gradients(zip(grads, q2.trainable_variables))

            with tf.GradientTape() as tape:
                tape.watch(q1.trainable_variables)
                predictions = q1(np.expand_dims(np.array(actions_did),axis=1))
                next_predictions= np.squeeze(predictions_l2.numpy())
                next_greedy_predictions=[]
                for k in range(len(next_predictions)):
                    next_greedy_predictions.append(np.max(next_predictions[k]))
                loss_sum = tf.keras.losses.MSE(np.expand_dims(np.array(next_greedy_predictions),axis=1),predictions)
                loss = tf.reduce_mean(loss_sum)
                grads = tape.gradient(loss, q1.trainable_variables)
                optimizer_critic.apply_gradients(zip(grads, q1.trainable_variables))
                loss_ev.append(np.squeeze(loss.numpy()))
    #
            with tf.GradientTape() as tape:
                actions = actor(np.expand_dims(np.zeros(batch_size),axis=1))
                tape.watch(actions)
                qvals = q1(actions)
            dq_da = tape.gradient(qvals, actions)

            with tf.GradientTape() as tape:
                tape.watch(actor.trainable_variables)
                actions = actor(np.expand_dims(np.zeros(batch_size),axis=1))
                theta = actor.trainable_variables
            da_dtheta = tape.gradient(actions, theta, output_gradients=-dq_da)
            optimizer_actor.apply_gradients(zip(da_dtheta, actor.trainable_variables))

    rtsum = rts/np.arange(1,len(rts)+1)
    if predict_q_table:
        predictions = np.squeeze(q1(np.expand_dims(np.array(betas_test),axis=1)).numpy())
        history = []
        history_1 = []

        for b in betas_test:
            history.append([[b, 0.]])
            history_1.append([[b,1.]])
        pred_guess_0 = np.squeeze(q2(np.expand_dims(np.array(history), axis=1)).numpy())
        pred_guess_1 = np.squeeze(q2(np.expand_dims(np.array(history_1), axis=1)).numpy())
        pred_guess = [pred_guess_0, pred_guess_1]

    plot_evolution_vKennedy_last(rt=rtsum, pt=pt, optimal=optimal, betas=betas_test,
     preds=predictions , pred_guess = pred_guess, loss=loss_ev, history_betas=history_betas, history_would_have_done=history_would_have_done,
      run_id= run_id)
    #
    save_models(run_id, models=[q1,q2,actor])
    data = "buffer_size: {}\Batch size: {}\nLearning_rate: {}\nOptimizer Critic: {}\nOptimizer Actor: {}".format(str(buffer_size), str(batch_size), str(lr),optimizer_critic.__str__(),optimizer_actor.__str__()  )
    #
    os.chdir(run_id)
    with open("info.txt", 'w') as f:
        f.write(data)
        f.close()
    os.chdir("..")

    return


check_folder("mate")

run_id=record()
number_run = "run_"+str(run_id)
real_training(run_id=number_run, lr=1e-4,
 states_wasted=10**5, batch_size=1000, buffer_size=2*10**3)



run_id=record()
number_run = "run_"+str(run_id)
real_training(run_id=number_run, lr=1e-5,
 states_wasted=10**5, batch_size=1000, buffer_size=5*10**3)


run_id=record()
number_run = "run_"+str(run_id)
real_training(run_id=number_run, lr=1e-5,
 states_wasted=3*10**5, batch_size=10**4, buffer_size=10**5)
