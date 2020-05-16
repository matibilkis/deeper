import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm as tqdm
tf.keras.backend.set_floatx('float32')
from collections import deque
from datetime import datetime
import random
import matplotlib
from environment import Environment

# from plots import *
from misc import *
from nets import *
from buffer import ReplayBuffer

def optimization_step(experiences, critic, critic_target, actor, actor_target, optimizer_critic, optimizer_actor):
    experiences = experiences.astype(np.float32)
    targeted_experience = actor_target.process_sequence_of_experiences(experiences)
    sequences, zeroed_rews = critic_target.process_sequence(targeted_experience)
    labels_critic = critic_target.give_td_error_Kennedy_guess( sequences, zeroed_rews)
    #
    ###### train the critic ######
    with tf.GradientTape() as tape:
        tape.watch(critic.trainable_variables)
        preds_critic = critic(sequences)
        loss_critic = tf.keras.losses.MSE(labels_critic, preds_critic)
        loss_critic = tf.reduce_mean(loss_critic)
        grads = tape.gradient(loss_critic, critic.trainable_variables)
        optimizer_critic.apply_gradients(zip(grads, critic.trainable_variables))
        loss_critic = np.squeeze(loss_critic.numpy())
    #
    #
    actor.lstm.reset_states()
    actor.lstm.stateful=False ### this is because the mask has trouble with differing the batch_size


    actions_indexed = [0.]*(actor.dolinar_layers)
    with tf.GradientTape() as tape:
        ##### get the actions only ######
        actions_with_outcomes = experiences.copy()
        act_ind=0
        for ind in range(experiences.shape[1]): #experiences.shape[0] = 2L +2
            if (ind%2 == 0)&(ind < 2*actor.dolinar_layers):
                ac = tf.convert_to_tensor(np.reshape(experiences[:,ind], (len(experiences),1,1)))
                actions_indexed[act_ind] = ac
                act_ind+=1
        
        actions_indexed = tf.concat(actions_indexed,axis=1)
        tape.watch(actions_indexed) ####watch the ations

        ### now prepare the state acions to put them into the critic###
        padded_data = [tf.ones((experiences.shape[0],1))*actor.pad_value]
        watched_input_critic  = padded_data.copy()
        ind_actions=0
        for ind,k in enumerate(tf.unstack(tf.convert_to_tensor(experiences[:,:-1]),axis=1)):
            if (ind%2==0)&(ind < 2*actor.dolinar_layers):
                padded_data.append(actions_indexed[:,ind_actions]) ### i add the input of the critic the watched actions!
                ind_actions+=1
            else:
                padded_data.append(tf.expand_dims(k, axis=1))
            if ind == 0:
                watched_input_critic = tf.stack([padded_data[0], padded_data[1]], axis=2) #importantly i put the padd first (state_action.)
            if (ind%2 == 0)&(ind!=0):
                intermediate = tf.stack([padded_data[ind], padded_data[ind+1]], axis=2)
                watched_input_critic = tf.concat([watched_input_critic, intermediate], axis=1)

        qvals = critic(watched_input_critic)
        dq_da = tape.gradient(qvals, actions_indexed)

    with tf.GradientTape() as tape:

        pads = np.ones(len(experiences)).astype(np.float32)*actor.pad_value
        news = np.random.rand(experiences.shape[0], experiences.shape[1]+1).astype(np.float32)
        news[:,1:] = experiences
        news[:,0] = pads
        instances_actor = [i for i in range(0,2*actor.dolinar_layers,2)]
        actionss = actor(np.reshape(news[:,instances_actor], (experiences.shape[0],actor.dolinar_layers,1)).astype(np.float32))

        da_dtheta = tape.gradient(actionss, actor.trainable_variables, output_gradients=-dq_da)

    #
    optimizer_actor.apply_gradients(zip(da_dtheta, actor.trainable_variables))
    actor.lstm.stateful=True
    return loss_critic



def ddpgKennedy(special_name="", dolinar_layers=2, number_phases=2, total_episodes = 10**3,buffer_size=500, batch_size=64, ep_guess=0.01,
 noise_displacement=0.5, lr_actor=0.01, lr_critic=0.001, tau=0.005):

    if not os.path.exists("results"):
        os.makedirs("results")

    env = Environment(amplitude=0.4, layers = dolinar_layers)
    buffer = ReplayBuffer(buffer_size=buffer_size)

    critic = Critic(nature="primary",valreg=0.01, dolinar_layers = dolinar_layers, number_phases=number_phases)
    critic_target = Critic(nature="target", dolinar_layers = dolinar_layers, number_phases=number_phases)
    actor = Actor(nature="primary", dolinar_layers = dolinar_layers)
    actor_target = Actor(nature="target", dolinar_layers = dolinar_layers)

    optimizer_critic = tf.keras.optimizers.Adam(lr=lr_critic)
    optimizer_actor = tf.keras.optimizers.Adam(lr=lr_actor) #0.001 works well


    rt = []
    pt = []
    new_loss=0.


    ##### STORING FOLDER ####
        ##### STORING FOLDER ####
            ##### STORING FOLDER ####
    numb = record()
    current_run_and_time ="results/run_" + str(numb)


    directory = current_run_and_time
    info_optimizers = "optimizer_critic_guess: {} \nOptimizer_actor_l0: {}\n".format(optimizer_critic.get_config(), optimizer_actor.get_config())
    infor_buffer = "Buffer_size: {}\n Batch_size for sampling: {}\n".format(buffer.buffer_size, batch_size)
    info_epsilons= "epsilon-guess: {}\nepsilon_displacement_noise: {}".format(ep_guess,noise_displacement)

    data = "tau: {}".format(tau) + "\n \n**** optimizers ***\n"+info_optimizers+"\n\n\n*** BUFFER ***\n"+infor_buffer+"\n\n\n *** NOISE PARAMETERS *** \n"+info_epsilons
    with open(directory+"/info.txt", 'w') as f:
        f.write(data)
        f.close()

    print("Beggining to train! \n \n")
    print(data)
    print("starting time: {}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    print("saving results in " + str(directory))
    avg_train = []
    ##### STORING FOLDER ####
        ##### STORING FOLDER ####
            ##### STORING FOLDER ####

    # history_betas = [] #to put in histogram
    # history_betas_would_have_done={"first":{}, "second":{"[0,0]":[]}} #to put in histogram
    # histo_preds = {"layer0":{}, "layer1":{}} #here i save the predictions to plot in a "straightforward way"

    actions = {}
    for layer in range(dolinar_layers+1):
        actions[str(layer)] = {}

    for k in outcomes_universe(dolinar_layers):
        for layer in range(dolinar_layers+1):
            actions[str(layer)][str(k[:layer])] = []

    #######
    at = make_attenuations(dolinar_layers)
    for episode in tqdm(range(total_episodes)):

        env.pick_phase()
        experiences=[] #where the current history of the current episode is stored
        context_outcome_actor = np.reshape(np.array([actor.pad_value]),(1,1,1)).astype(np.float32)
        for layer in range(actor.dolinar_layers):
            print(layer)
            beta_would_do = np.squeeze(actor(context_outcome_actor))
            beta =  beta_would_do + np.random.uniform(-noise_displacement, noise_displacement)
            outcome = env.give_outcome(beta,layer)
            experiences.append(beta)
            experiences.append(outcome)

            context_outcome_actor = np.reshape(np.array([outcome]),(1,1,1)).astype(np.float32)

        ### ep-gredy guessing of the phase###
        if np.random.random()< ep_guess:
            val = np.random.choice(range(number_phases),1)[0]
            guess_index, guess_input_network = val, val/critic.number_phases
        else:
            guess_index, guess_input_network = critic.give_favourite_guess(experiences) #experiences is the branch of the current tree of actions + outcomes.
        experiences.append(guess_input_network)

        reward = env.give_reward(guess_index)
        experiences.append(reward)
        # # reward = qval(beta, outcome, guess)
        buffer.add(tuple(experiences))

        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######

        if buffer.count>1:
            sampled_experiences = buffer.sample(batch_size)
            new_loss = optimization_step(sampled_experiences, critic, critic_target, actor, actor_target, optimizer_critic, optimizer_actor)
            critic_target.update_target_parameters(critic)
            actor_target.update_target_parameters(actor)
#####
        avg_train.append(new_loss)
        # avg_test.append(test_loss.result().numpy())
    #
        rt.append(reward)
    #
        # actor.lstm.reset_states()
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
    tau = .05
    lr_critic = 0.0001
    lr_actor=0.001
    noise_displacement = .5
    batch_size = 128
    ep_guess=0.01

    name_run = ddpgKennedy(total_episodes=2000, dolinar_layers=2, noise_displacement=noise_displacement, tau=tau,
    buffer_size=2*10**6, batch_size=batch_size, lr_critic=lr_critic, lr_actor=lr_actor, ep_guess=ep_guess)

    info_run +="***\n***\nname_run: {} ***\ntau: {}\nlr_critic: {}\nnoise_displacement: {}\nbatch_size: {}\n-------\n-------\n\n".format(name_run,tau, lr_critic, noise_displacement, batch_size)

    to_csv.append({"name_run":"run_"+str(name_run), "tau": tau, "lr_critic":lr_critic, "noise_displacement": noise_displacement,
    "BS":batch_size})

    with open("results/info_runs.txt", 'a+') as f:
        f.write(info_run)
        f.close()
    ##### if we put more runs... ###
    # pp = pd.DataFrame(to_csv)
    # pp.to_csv("results/panda_info.csv")
