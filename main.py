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
from misc import Prob, ps_maxlik, qval
from nets import *
from buffer import ReplayBuffer


def ddpgKennedy(name_run="",total_episodes = 10**3,buffer_size=500, batch_size=64, ep_guess=0.01,
 noise_displacement=0.5,lr_actor=0.01, lr_critic=0.001, tau=0.005, repetitions=1):

    amplitude = 0.4
    buffer = ReplayBuffer(buffer_size=buffer_size)

    critic_q0 = Critic(input_dim=1)
    actor_q0 = Actor(input_dim=1)
    critic_guess = Critic(input_dim=3)
    target_guess = Critic(input_dim=3)

    critic_q0(np.array([[0.],[1.]])) #initialize the network 0, arbitrary inputs.
    actor_q0(np.array([[0.],[1.]])) #initialize the network 0, arbitrary inputs.
    critic_guess(np.array([[0.,1.,1.]]))
    target_guess(np.array([[0.,1.,1.]]))
    #
    optimizer_critic_guess = tf.keras.optimizers.Adam(lr=lr_critic)
    optimizer_actor_l0 = tf.keras.optimizers.Adam(lr=lr_actor)
    optimizer_critic_l0 = tf.keras.optimizers.Adam(lr=lr_actor)


    rt = []
    pt = []

    #define this global so i use them in a function defined above... optimizatin step and testing()
    train_loss_l0 = tf.keras.metrics.Mean('train_loss_l0', dtype=tf.float32)
    test_loss_l0 = tf.keras.metrics.Mean('test_loss_l0', dtype=tf.float32)
    train_loss_l1 = tf.keras.metrics.Mean('train_loss_l1', dtype=tf.float32)
    test_loss_l1 = tf.keras.metrics.Mean('test_loss_l1', dtype=tf.float32)


    if name_run == "":
        current_run_and_time = "results/{}".format(datetime.now().strftime("%Y%m%d-%H%M"))
    else:
        current_run_and_time = "results/"+name_run
    directory = current_run_and_time
    train_log_dir_0 =  current_run_and_time + '/train_l0'
    test_log_dir_0 =   current_run_and_time + '/test_l0'
    train_log_dir_1 =   current_run_and_time + '/train_l1'
    test_log_dir_1 =   current_run_and_time + '/test_l1'
    train_summary_writer_0 = tf.summary.create_file_writer(train_log_dir_0)
    train_summary_writer_1 = tf.summary.create_file_writer(train_log_dir_1)
    test_summary_writer_0 = tf.summary.create_file_writer(test_log_dir_0)
    test_summary_writer_1 = tf.summary.create_file_writer(test_log_dir_1)

    info_optimizers = "optimizer_critic_guess: {} \nOptimizer_actor_l0: {}\nOptimizer_critic_l0: {}\n".format(optimizer_critic_guess.get_config(), optimizer_actor_l0.get_config(), optimizer_critic_l0)
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
    avg_train_l0 = []
    avg_train_l1 = []
    avg_test_l0 = []
    avg_test_l1 = []

    history_betas = [] #to put in histogram
    history_betas_would_have_done=[] #to put in histogram
    histo_preds = {"net_0":{}, "net_1":{}} #here i save the predictions to plot in a "straightforward way"

    #######
    for episode in tqdm(range(total_episodes)):

        alice_phase = np.random.choice([-1.,1.],1)[0]
        beta_would_do = actor_q0(np.array([[0.]])).numpy()[0][0]
        beta =  beta_would_do + np.random.uniform(-noise_displacement, noise_displacement)
        proboutcome = Prob(alice_phase*amplitude,beta,0)
        outcome = np.random.choice([0.,1.],1,p=[proboutcome, 1-proboutcome])[0]

        history_betas.append(beta)
        history_betas_would_have_done.append(beta_would_do)
        #epsilon-greedy choice of the guessing! Do you imagine other way to do this? How would you apply UCB ? discretize?
        if np.random.random()< ep_guess:
            guess = np.random.choice([-1.,1.],1)[0]
        else:
            guess = critic_guess.give_favourite_guess(beta, outcome)
        if guess == alice_phase:
            reward = 1.
        else:
            reward = 0.
        buffer.add(beta, outcome, guess, reward)


        ### optimization step and testing the generalization performance ! ####
        optimization_step(networks=[actor_q0, critic_q0, critic_guess, target_guess],
                          optimizers = [optimizer_critic_guess,  optimizer_actor_l0, optimizer_critic_l0 ],
                          losses=[test_loss_l0, test_loss_l1],
                          buffer=buffer,
                          batch_size=batch_size,repetitions=repetitions)
        testing_data(losses=[train_loss_l0, train_loss_l1], buffer=buffer, networks=[actor_q0, critic_q0, critic_guess, target_guess])

        ### i append the losses to plot them later ###
        avg_train_l0.append(train_loss_l0.result().numpy())
        avg_train_l1.append(train_loss_l1.result().numpy())
        avg_test_l0.append(test_loss_l0.result().numpy())
        avg_test_l1.append(test_loss_l1.result().numpy())


        ### appending the reward to calculate cumulative! ###
        rt.append(reward)

        ### calculate success probability if the agent went greedy ###
        p=0
        for outcome in [0.,1.]:
            p+=Prob(critic_guess.give_favourite_guess(beta_would_do, outcome)*amplitude, beta_would_do,outcome)
        p/=2
        pt.append(p)



        with train_summary_writer_0.as_default():
            tf.summary.scalar('loss', train_loss_l0.result(), step=episode)
        with test_summary_writer_0.as_default():
            tf.summary.scalar('loss', test_loss_l0.result(), step=episode)
        with train_summary_writer_1.as_default():
            tf.summary.scalar('loss', train_loss_l1.result(), step=episode)
        with test_summary_writer_1.as_default():
            tf.summary.scalar('loss', test_loss_l1.result(), step=episode)

        if episode%(total_episodes/10) == 0: #this is for showing 10 results in total.

            template = 'Episode {}, \Rt: {}, \Pt: {}, Train loss_l1: {}, Test loss_l1: {}, Train Loss_l0: {}, Test Loss_l0: {}\n\n'
            print(template.format(episode+1,
                                np.sum(rt)/(episode+1),
                                  pt[-1],
                                 np.round(train_loss_l1.result().numpy(),5),
                                 test_loss_l1.result().numpy(),
                                 np.round(train_loss_l0.result().numpy(),15),
                                 np.round(test_loss_l0.result().numpy(),5))
                  )

            for nett in ["net_0","net_1"]: #net_0 will be critic_q0, net_1 will be critic_qguess

                histo_preds[nett][str(episode)] ={}
                histo_preds[nett][str(episode)]["episode"] = episode
                histo_preds[nett][str(episode)]["values"] = {}

                histo_preds["net_0"][str(episode)]["values"] = np.squeeze(critic_q0(np.expand_dims(buffer.betas,axis=1)))

            index=0
            for n1 in [0.,1.]:
                for guess in [-1.,1.]:
                    foo =np.array([[b,n1,guess] for b in buffer.betas]) #betas_train defined as global in create_dataset_l2()
                    histo_preds["net_1"][str(episode)]["values"][str(index)] = np.squeeze(critic_guess(foo))
                    index+=1


    rt = [np.sum(rt[:k]) for k in range(len(rt))]
    rt = rt/np.arange(1,len(rt)+1)
    losses = [[avg_train_l0, avg_test_l0], [ avg_train_l1, avg_test_l1]]
    BigPlot(buffer,rt, pt, history_betas, history_betas_would_have_done, histo_preds, losses, directory)
    plot_inside_buffer(buffer, directory)
    return #rt, pt, history_betas, history_betas_would_have_done, histo_preds, losses, name_directory-


if __name__ == "__main__":
    info_run = ""
    for tau in [0.1, 0.01, 0.005]:
        for lr_critic in [.001, .01]:
            for noise_displacement in [1, 0.1, 0.001]:
                for batch_size in [32., 256., 512.]:
                    name_run = datetime.now().strftime("%m-%d-%H-%-M%-S")
                    info_run +="***\n***\nname_run: {} ***\ntau: {}\nlr_critic: {}\nnoise_displacement: {}\nbatch_size: {}\n-------\n-------\n\n".format(name_run,tau, lr_critic, noise_displacement, batch_size)

        ddpgKennedy(name_run, total_episodes=10**4, noise_displacement=noise_displacement, tau=tau,
            buffer_size=10**5, batch_size=batch_size, lr_critic=lr_critic, lr_actor=0.001)

    # if os.path.isfile("results/info_runs.txt") == True:
    #
    with open("results/info_runs.txt", 'a+') as f:
        f.write(info_run)
        f.close()
