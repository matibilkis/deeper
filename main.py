import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm as tqdm
tf.keras.backend.set_floatx('float64')
from collections import deque
from datetime import datetime
import random
import matplotlib

from plots import *
from misc import Prob, ps_maxlik, qval, record
from nets import *
from buffer import ReplayBuffer


def ddpgKennedy(special_name="",total_episodes = 10**3,buffer_size=500, batch_size=64, ep_guess=0.01,
 noise_displacement=0.5,lr_actor=0.01, lr_critic=0.001, tau=0.005, repetitions=1, plots=True):

    if not os.path.exists("results"):
        os.makedirs("results")

    amplitude = 0.4
    buffer = ReplayBuffer(buffer_size=buffer_size)

    critic = Critic()
    critic_target = Critic()
    actor = Actor(input_dim=1)

    actor(np.array([[0.]]).astype(np.float32)) #initialize the network 0, arbitrary inputs.
    #
    optimizer_critic = tf.keras.optimizers.Adam(lr=lr_critic)
    optimizer_actor = tf.keras.optimizers.Adam(lr=lr_actor)


    rt = []
    pt = []

    #define this global so i use them in a function defined above... optimizatin step and testing()
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)


    if special_name == "":
        # current_run_and_time = "results/{}".format(datetime.now().strftime("%Y%m%d-%H%M"))
        numb = record()
        current_run_and_time ="results/run_" + str(numb)
    else:
        current_run_and_time = "results/"+special_name

    directory = current_run_and_time
    train_log =  current_run_and_time + '/train_l0'
    test_log =   current_run_and_time + '/test_l0'

    train_summary_writer = tf.summary.create_file_writer(train_log)
    test_summary_writer_0 = tf.summary.create_file_writer(test_log)

    info_optimizers = "optimizer_critic_guess: {} \nOptimizer_actor_l0: {}\n".format(optimizer_critic.get_config(), optimizer_actor.get_config())
    infor_buffer = "Buffer_size: {}\n Batch_size for sampling: {}\n".format(buffer.buffer_size, batch_size)
    info_epsilons= "epsilon-guess: {}\nepsilon_displacement_noise: {}".format(ep_guess,noise_displacement)

    data = "tau: {}, repetitions per optimization step (would be like epochs): {}".format(tau,repetitions) + "\n \n**** optimizers ***\n"+info_optimizers+"\n\n\n*** BUFFER ***\n"+infor_buffer+"\n\n\n *** NOISE PARAMETERS *** \n"+info_epsilons
    with open(directory+"/info.txt", 'w') as f:
        f.write(data)
        f.close()

    print("Beggining to train! \n \n")
    print(data)
    print("starting time: {}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    print("saving results in " + str(directory))
    avg_train = []
    avg_test = []

    history_betas = [] #to put in histogram
    history_betas_would_have_done=[] #to put in histogram
    histo_preds = {"critic":{}} #here i save the predictions to plot in a "straightforward way"

    #######
    for episode in tqdm(range(total_episodes)):

        alice_phase = np.random.choice([-1.,1.],1)[0]
        beta_would_do = actor(np.array([[0.]])).numpy()[0][0]
        beta =  beta_would_do + max(0.1, np.random.uniform(-noise_displacement, noise_displacement)*np.exp(-episode/300))
        proboutcome = Prob(alice_phase*amplitude,beta,0)
        outcome = np.random.choice([0.,1.],1,p=[proboutcome, 1-proboutcome])[0]

        history_betas.append(beta)
        history_betas_would_have_done.append(beta_would_do)

    #
        if np.random.random()< ep_guess:
            guess = np.random.choice([-1.,1.],1)[0]
        else:
            sequence = np.array([[ [beta, critic.pad_value], [outcome, -1.]]  ]).astype(np.float32)
            guess = critic.give_favourite_guess(sequence)
        if guess == alice_phase:
            reward = 1.
        else:
            reward = 0.
        buffer.add(beta, outcome, guess, reward)


        ###### END OF OPTIMIZATION STEP ######
        ###### END OF OPTIMIZATION STEP ######
        experiences = buffer.sample(batch_size)
        sequences, zeroed_rews = critic.process_sequence(experiences)
        labels_critic = critic_target.give_td_error_Kennedy_guess( sequences, zeroed_rews)
        with tf.GradientTape() as tape:
            tape.watch(critic.trainable_variables)
            preds_critic = critic(sequences)
            loss_critic = tf.keras.losses.MSE(labels_critic, preds_critic)
            loss_critic = tf.reduce_mean(loss_critic)
            grads = tape.gradient(loss_critic, critic.trainable_variables)
            optimizer_critic.apply_gradients(zip(grads, critic.trainable_variables))
            train_loss(loss_critic)

        critic_target.update_target_parameters(critic, tau=tau)

        with tf.GradientTape() as tape:
            ones = tf.ones(shape=(experiences.shape[0],1))
            actions = tf.cast(actor(np.expand_dims(np.zeros(len(experiences)),axis=1)), tf.float32)   #This can be improved i think!! (the conversion... )

            tape.watch(actions)
            qvals = critic(tf.expand_dims(tf.concat([actions, ones], axis=1),axis=1))
            dq_da = tape.gradient(qvals, actions)

        with tf.GradientTape() as tape:
            actionss = tf.cast(actor(np.expand_dims(np.zeros(len(experiences)),axis=1)), tf.float32)
            da_dtheta = tape.gradient(actionss, actor.trainable_variables, output_gradients=-dq_da)

        optimizer_actor.apply_gradients(zip(da_dtheta, actor.trainable_variables))
        ###### END OF OPTIMIZATION STEP ######
        ###### END OF OPTIMIZATION STEP ######

        avg_train.append(train_loss.result().numpy())
        avg_test.append(test_loss.result().numpy())
    #
        rt.append(reward)
    #

        ########################################################################
        ### calculate success probability if the agent went greedy ###########
        p=0
        for outcome in [0.,1.]:
            p+=Prob(critic.give_favourite_guess(critic.pad_single_sequence([beta_would_do, outcome, -1.]))*amplitude, beta_would_do,outcome)
        p/=2
        pt.append(p)
        ################


    rt = [np.sum(rt[:k]) for k in range(len(rt))]
    rt = rt/np.arange(1,len(rt)+1)

    losses = [avg_train, avg_test]

    BigPlot(buffer,rt, pt, history_betas, history_betas_would_have_done, histo_preds, losses, directory)

# ###### UP TO HERE !!! #####
#         ### optimization step and testing the generalization performance ! ####
#         optimization_step(networks=[actor, critic, target_critic],
#                           optimizers = [optimizer_critic, optimizer_actor],
#                           losses=[train_loss, test_loss],
#                           buffer=buffer,
#                           batch_size=batch_size,
#                           repetitions=repetitions)
#         testing_data(losses=[train_loss_l0, train_loss_l1], buffer=buffer, networks=[actor, critic])
#
#         ### i append the losses to plot them later ###
#         avg_train.append(train_loss.result().numpy())
#         avg_test.append(test_loss.result().numpy())
#
#         ### appending the reward to calculate cumulative! ###
#         rt.append(reward)
#
#         ### calculate success probability if the agent went greedy ###
#         p=0
#         for outcome in [0.,1.]:
#             p+=Prob(critic_guess.give_favourite_guess(beta_would_do, outcome)*amplitude, beta_would_do,outcome)
#         p/=2
#         pt.append(p)
#
#
#
#         # with train_summary_writer_0.as_default():
#         #     tf.summary.scalar('loss', train_loss.result(), step=episode)
#         # with test_summary_writer_0.as_default():
#         #     tf.summary.scalar('loss', test_loss_l0.result(), step=episode)
#         # with train_summary_writer_1.as_default():
#         #     tf.summary.scalar('loss', train_loss_l1.result(), step=episode)
#         # with test_summary_writer_1.as_default():
#         #     tf.summary.scalar('loss', test_loss_l1.result(), step=episode)
#
#         if episode%(total_episodes/10) == 0: #this is for showing 10 results in total.
#
#             template = 'Episode {}, \Rt: {}, \Pt: {}, Train loss: {}, Test loss: {}\n\n'
#             print(template.format(episode+1,
#                                 np.sum(rt)/(episode+1),
#                                   pt[-1],
#                                  np.round(train_loss.result().numpy(),5),
#                                  test_loss_l1.result().numpy(),
#                                  np.round(test_loss.result().numpy(),5))
#                   )
#
#             for nett in ["net_0","net_1"]: #net_0 will be critic_q0, net_1 will be critic_qguess
#
#                 histo_preds[nett][str(episode)] ={}
#                 histo_preds[nett][str(episode)]["episode"] = episode
#                 histo_preds[nett][str(episode)]["values"] = {}
#
#                 histo_preds["net_0"][str(episode)]["values"] = np.squeeze(critic_q0(np.expand_dims(buffer.betas,axis=1)))
#
#             index=0
#             for n1 in [0.,1.]:
#                 for guess in [-1.,1.]:
#                     foo =np.array([[b,n1,guess] for b in buffer.betas]) #betas_train defined as global in create_dataset_l2()
#                     histo_preds["net_1"][str(episode)]["values"][str(index)] = np.squeeze(critic_guess(foo))
#                     index+=1
#
#
#     rt = [np.sum(rt[:k]) for k in range(len(rt))]
#     rt = rt/np.arange(1,len(rt)+1)
#     losses = [[avg_train_l0, avg_test_l0], [ avg_train_l1, avg_test_l1]]
#     if plots:
#         BigPlot(buffer,rt, pt, history_betas, history_betas_would_have_done, histo_preds, losses, directory)
#         plot_inside_buffer(buffer, directory)
#     return "run_"+str(numb)#rt, pt, history_betas, history_betas_would_have_done, histo_preds, losses, name_directory-
    return


if __name__ == "__main__":
    info_run = ""
    to_csv=[]
    for tau in [0.01]:
        for lr_critic in [.01]:
            for noise_displacement in [0.1]:
                for batch_size in [ 8.]:

                    # name_run = datetime.now().strftime("%m-%d-%H-%-M%-S")

                    name_run = ddpgKennedy(total_episodes=10**3, noise_displacement=noise_displacement, tau=tau,
                    buffer_size=10**3, batch_size=batch_size, lr_critic=0.01, lr_actor=0.01, plots=True)

                    info_run +="***\n***\nname_run: {} ***\ntau: {}\nlr_critic: {}\nnoise_displacement: {}\nbatch_size: {}\n-------\n-------\n\n".format(name_run,tau, lr_critic, noise_displacement, batch_size)

                    to_csv.append({"name_run":"run_"+str(name_run), "tau": tau, "lr_critic":lr_critic, "noise_displacement": noise_displacement,
                    "BS":batch_size})


    # if os.path.isfile("results/info_runs.txt") == True:
    #
    with open("results/info_runs.txt", 'a+') as f:
        f.write(info_run)
        f.close()

    pp = pd.DataFrame(to_csv)
    pp.to_csv("results/panda_info.csv")
