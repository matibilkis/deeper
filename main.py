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

# from plots import *
from misc import *
from nets import *
from buffer import ReplayBuffer

def optimization_step(experiences, critic, critic_target, actor, actor_target, optimizer_critic, optimizer_actor, train_loss):
    targeted_experience = actor_target.targeted_sequence(experiences)
    sequences, zeroed_rews = critic_target.process_sequence(targeted_experience)
    labels_critic = critic_target.give_td_error_Kennedy_guess( sequences, zeroed_rews)
    #
    # with tf.GradientTape() as tape:
    #     tape.watch(critic.trainable_variables)
    #     preds_critic = critic(sequences)
    #     loss_critic = tf.keras.losses.MSE(labels_critic, preds_critic)
    #     loss_critic = tf.reduce_mean(loss_critic)
    #     grads = tape.gradient(loss_critic, critic.trainable_variables)
    #     optimizer_critic.apply_gradients(zip(grads, critic.trainable_variables))
    #     train_loss(loss_critic)
    #
    #
    # with tf.GradientTape() as tape:
    #     ones = tf.ones(shape=(experiences.shape[0],1))*critic.pad_value
    #     actions = actor(np.expand_dims(np.zeros(len(experiences)),axis=1))   #This can be improved i think!! (the conversion... )
    #
    #     tape.watch(actions)
    #     qvals = critic(tf.expand_dims(tf.concat([actions, ones], axis=1),axis=1))
    #     dq_da = tape.gradient(qvals, actions)
    #
    # with tf.GradientTape() as tape:
    #     actionss = actor(np.expand_dims(np.zeros(len(experiences)),axis=1))
    #     da_dtheta = tape.gradient(actionss, actor.trainable_variables, output_gradients=-dq_da)
    #
    # optimizer_actor.apply_gradients(zip(da_dtheta, actor.trainable_variables))
    return 0.
        ###### END OF OPTIMIZATION STEP ######
    ###### END OF OPTIMIZATION STEP ######


def ddpgKennedy(special_name="",total_episodes = 10**3,buffer_size=500, batch_size=64, ep_guess=0.1,
 noise_displacement=0.5,lr_actor=0.01, lr_critic=0.001, tau=0.005, repetitions=1, plots=True):

    if not os.path.exists("results"):
        os.makedirs("results")

    amplitude = 0.4
    buffer = ReplayBuffer(buffer_size=buffer_size)

    critic = Critic(nature="primary",valreg=0.01)
    critic_target = Critic(nature="target")
    actor = Actor(nature="primary")
    critic_actor = Actor(nature="target")

    optimizer_critic = tf.keras.optimizers.Adam(lr=lr_critic)
    optimizer_actor = tf.keras.optimizers.Adam(lr=lr_actor) #0.001 works well


    rt = []
    pt = []
    new_loss=0.


        # current_run_and_time = "results/{}".format(datetime.now().strftime("%Y%m%d-%H%M"))
    numb = record()
    current_run_and_time ="results/run_" + str(numb)


    directory = current_run_and_time
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

    # history_betas = [] #to put in histogram
    # history_betas_would_have_done={"first":{}, "second":{"[0,0]":[]}} #to put in histogram
    # histo_preds = {"layer0":{}, "layer1":{}} #here i save the predictions to plot in a "straightforward way"

    actions = {}
    for layer in range(3):
        actions[str(layer)] = {}

    for k in outcomes_universe(2):
        for layer in range(3):
            actions[str(layer)][str(k[:layer])] = []

    #######
    at = make_attenuations(2)
    for episode in tqdm(range(total_episodes)):

        alice_phase = np.random.choice([-1.,1.],1)[0]
        input_actor = np.reshape(np.array([actor.pad_value]),(1,1,1))
        beta_would_do = np.squeeze(actor(input_actor))
        beta1 =  beta_would_do + np.random.uniform(-noise_displacement, noise_displacement)
        proboutcome = P(alice_phase*amplitude,beta1,np.cos(at[0]), 0.)
        outcome1 = np.random.choice([0.,1.],1,p=[proboutcome, 1-proboutcome])[0]


        input_actor = np.reshape(np.array([outcome1]),(1,1,1))
        beta_would_do = np.squeeze(actor(input_actor))
        beta2 =  beta_would_do + np.random.uniform(-noise_displacement, noise_displacement)
        proboutcome = P(alice_phase*amplitude,beta2,np.sin(at[0])*np.cos(at[0]), 0.) #it's 1, i know...
        outcome2 = np.random.choice([0.,1.],1,p=[proboutcome, 1-proboutcome])[0]
        #history_betas.append(beta)
        #history_betas_would_have_done.append(beta_would_do)

    #
        guess = np.random.choice([-1.,1.],1)[0]
        #
        # if np.random.random()< ep_guess:
        #     guess = np.random.choice([-1.,1.],1)[0]
        # else:
        #     guess = critic.give_favourite_guess(critic.pad_single_sequence([beta1, outcome1, beta2, outcome2, 1.]))
        if guess == alice_phase:
            reward = 1.
        else:
            reward = 0.
        # # reward = qval(beta, outcome, guess)
        buffer.add((beta1, outcome1, beta2, outcome2, guess, reward))


        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######

        if buffer.count>1:
            experiences = buffer.sample(batch_size)
            new_loss = optimization_step(experiences,critic, critic_target, actor, optimizer_critic, optimizer_actor)
            critic_target.update_target_parameters(critic, tau=tau)


#####
        avg_train.append(new_loss)
        # avg_test.append(test_loss.result().numpy())
    #
        rt.append(reward)
    #
        actor.lstm.reset_states()
        ########################################################################
        # ### calculate success probability if the agent went greedy ###########
        # p=0
        # for outcome in [0.,1.]:
        #     guess = critic.give_favourite_guess(critic.pad_single_sequence([beta_would_do, outcome, 1.]))
        #     # print(guess, outcome)
        #     p+=Prob(guess*amplitude, beta_would_do,outcome) #Notice it's very very important that the sequence has the 1. and not -1!!! TO DO in a better way!
        # p/=2
        pt.append(0)
        ################

        if episode%(total_episodes/10) == 0: #this is for showing 10 results in total.

            template = 'Episode {}, \Rt: {}, \Pt: {}, Train loss: {}\n\n'
            print(template.format(episode+1,
                                np.sum(rt)/(episode+1),
                                  pt[-1],
                                 np.round(np.array(avg_train).mean(),5),
                                )
                  )

            #
            # for layer in ["layer0","layer1"]: #net_0 will be critic_q0, net_1 will be critic_qguess
            #
            #     histo_preds[layer][str(episode)] ={}
            #     histo_preds[layer][str(episode)]["episode"] = episode
            #     histo_preds[layer][str(episode)]["values"] = {}
            #
            # simp = np.random.randn(len(buffer.betas),4)
            # simp[:,0] =buffer.betas
            # qvals0 = np.squeeze(critic(critic.process_sequence(simp)[0]).numpy()[:,0])
            # histo_preds["layer0"][str(episode)]["values"] = qvals0
            #
            # index=0
            # for n1 in [0.,1.]:
            #     for guess in [-1.,1.]:
            #         simp[:,1] = n1
            #         simp[:,2] = guess
            #         qvals1 = np.squeeze(critic(critic.process_sequence(simp)[0]).numpy()[:,1])
            #         histo_preds["layer1"][str(episode)]["values"][str(index)] = qvals1
            #         index+=1
            #


    rt = [np.sum(rt[:k]) for k in range(len(rt))]
    rt = rt/np.arange(1,len(rt)+1)

    # losses = [avg_train, avg_test]

    # BigPlot(buffer,rt, pt, history_betas, history_betas_would_have_done, histo_preds, losses, directory)
    return


if __name__ == "__main__":
    info_run = ""
    to_csv=[]
    for tau in [.5, 0.05]:
        for lr_critic in [0.0001]:
            for noise_displacement in [.5, .25, .1]:
                for batch_size in [ 64., 128.]:

                    # name_run = datetime.now().strftime("%m-%d-%H-%-M%-S")

                    name_run = ddpgKennedy(total_episodes=2000, noise_displacement=noise_displacement, tau=tau,
                    buffer_size=2*10**6, batch_size=batch_size, lr_critic=lr_critic, lr_actor=0.001, plots=True)

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
