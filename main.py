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
from plots import just_plot
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
        tape.watch(actor.trainable_variables)
        pads = np.ones(len(experiences)).astype(np.float32)*actor.pad_value
        news = np.random.rand(experiences.shape[0], experiences.shape[1]+1).astype(np.float32)
        news[:,1:] = experiences
        news[:,0] = pads
        instances_actor = [i for i in range(0,2*actor.dolinar_layers,2)]
        actionss = actor(np.reshape(news[:,instances_actor], (experiences.shape[0],actor.dolinar_layers,1)).astype(np.float32))
        da_dtheta = tape.gradient(actionss, actor.trainable_variables, output_gradients=-dq_da)

    # #
    # print(dq_da)
    # print("***LLLLL****")
    # print(da_dtheta)
    # print("*/!($/$!TY)R(G)")
    optimizer_actor.apply_gradients(zip(da_dtheta, actor.trainable_variables))
    actor.lstm.stateful=True
    return loss_critic



def RDPG(special_name="", amplitude=0.4, dolinar_layers=2, number_phases=2, total_episodes = 10**3, buffer_size=500, batch_size=64, ep_guess=0.01,
 noise_displacement=0.5, lr_actor=0.01, lr_critic=0.001, tau=0.005):

    if not os.path.exists("results"):
        os.makedirs("results")

    env = Environment(amplitude=amplitude, dolinar_layers = dolinar_layers, number_phases=number_phases)
    buffer = ReplayBuffer(buffer_size=buffer_size)

    critic = Critic(nature="primary",valreg=0.01, dolinar_layers = dolinar_layers, number_phases=number_phases)
    critic_target = Critic(nature="target", dolinar_layers = dolinar_layers, number_phases=number_phases)
    actor = Actor(nature="primary", dolinar_layers = dolinar_layers)
    actor_target = Actor(nature="target", dolinar_layers = dolinar_layers)

    optimizer_critic = tf.keras.optimizers.Adam(lr=lr_critic)
    optimizer_actor = tf.keras.optimizers.Adam(lr=lr_actor)

    policy_evaluator = PolicyEvaluator(amplitude = amplitude, dolinar_layers=dolinar_layers, number_phases = number_phases)

    rt = []
    pt = []
    new_loss=0.

    ##### STORING FOLDER ####
    ##### STORING FOLDER ####
    ##### STORING FOLDER ####
    numb = record()
    directory ="results/run_" + str(numb)


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

    # actions = {}
    # for layer in range(dolinar_layers+1):
    #     actions[str(layer)] = {}
    #
    # for k in outcomes_universe(dolinar_layers):
    #     for layer in range(dolinar_layers+1):
    #         actions[str(layer)][str(k[:layer])] = []

    #######
    at = make_attenuations(dolinar_layers)
    for episode in tqdm(range(total_episodes)):

        env.pick_phase()
        experiences=[] #where the current history of the current episode is stored
        context_outcome_actor = np.reshape(np.array([actor.pad_value]),(1,1,1)).astype(np.float32)
        for layer in range(actor.dolinar_layers):
            beta_would_do = np.squeeze(actor(context_outcome_actor))
            beta =  beta_would_do + np.random.uniform(-noise_displacement, noise_displacement)#np.clip(,-2*amplitude,2*amplitude)
            outcome = env.give_outcome(beta,layer)
            experiences.append(beta)
            experiences.append(outcome)
            context_outcome_actor = np.reshape(np.array([outcome]),(1,1,1)).astype(np.float32)

        ### ep-gredy guessing of the phase###
        ### ep-gredy guessing of the phase###
        if np.random.random()< ep_guess:
            val = np.random.choice(range(number_phases),1)[0]
            guess_index, guess_input_network = val, val/critic.number_phases
        else:
            guess_index, guess_input_network = critic.give_favourite_guess(experiences) #experiences is the branch of the current tree of actions + outcomes.
        experiences.append(guess_input_network)

        reward = env.give_reward(guess_index)
        experiences.append(reward)
        buffer.add(tuple(experiences))

        actor.lstm.reset_states()
        actor.lstm.stateful=False

        rt.append(reward)
        pt.append(policy_evaluator.greedy_strategy(actor = actor, critic = critic))

        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######
        if (buffer.count>100):#(episode%100==1):
            sampled_experiences = buffer.sample(batch_size)
            new_loss = optimization_step(sampled_experiences, critic, critic_target, actor, actor_target, optimizer_critic, optimizer_actor)
            critic_target.update_target_parameters(critic)
            actor_target.update_target_parameters(actor)
            noise_displacement = max(0.1,0.999*noise_displacement)
        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######

        avg_train.append(new_loss)
        #actor.lstm.reset_states()
        actor.lstm.stateful=True
        actor.lstm.reset_states()

        if episode%(total_episodes/100) == 0: #this is for showing 10 results in total.

            template = 'Episode {}, \Rt: {}, \Pt: {}, Train loss: {}\n\n'
            print(template.format(episode+1,
                                np.sum(rt)/(episode+1),
                                  pt[-1],
                                 np.round(np.array(avg_train).mean(),5),
                                )
                  )

    cumre=0
    rrt = []
    for k in rt:
        cumre+=k
        rrt.append(cumre)
    rrt = rrt/np.arange(1,len(rt)+1)

    np.save(directory+"/learning_curves/", rrt)
    np.save(directory+"/learning_curves/", pt)

    for model, net_folder in zip([actor, actor_target, critic, critic_target],["actor_primary", "actor_target", "critic_primary", "critic_target"]):
        model.save_weights(directory+"/networks/"+net_folder+"/")
    just_plot(rrt, pt, avg_train, env.helstrom(), directory)
    # BigPlot(buffer,rt, pt, history_betas, history_betas_would_have_done, histo_preds, losses, directory)
    return


if __name__ == "__main__":
    info_run = ""
    to_csv=[]
    amplitude=0.4
    tau = .01
    lr_critic = 0.01
    lr_actor=0.01
    noise_displacement = .25
    batch_size = 128.
    ep_guess=0.01
    dolinar_layers=2
    number_phases=2


    name_run = RDPG(amplitude=amplitude, total_episodes=10**4, dolinar_layers=dolinar_layers, noise_displacement=noise_displacement, tau=tau,
    buffer_size=10**8, batch_size=batch_size, lr_critic=lr_critic, lr_actor=lr_actor, ep_guess=ep_guess)

    info_run +="***\n***\nname_run: {} ***\ntau: {}\nlr_critic: {}\nnoise_displacement: {}\nbatch_size: {}\n-------\n-------\n\n".format(name_run,tau, lr_critic, noise_displacement, batch_size)

    to_csv.append({"name_run":"run_"+str(name_run), "tau": tau, "lr_critic":lr_critic, "noise_displacement": noise_displacement,
    "BS":batch_size})

    with open("results/info_runs.txt", 'a+') as f:
        f.write(info_run)
        f.close()
    ##### if we put more runs... ###
    # pp = pd.DataFrame(to_csv)
    # pp.to_csv("results/panda_info.csv")
