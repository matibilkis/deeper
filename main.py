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



@tf.function
def step_critic_tf(batched_input,labels_critic, critic, optimizer_critic):
    with tf.GradientTape() as tape:
        tape.watch(critic.trainable_variables)
        preds_critic = critic(batched_input)
        loss_critic = tf.keras.losses.MSE(tf.expand_dims(labels_critic, axis=2), preds_critic)
        loss_critic = tf.reduce_mean(loss_critic)
        grads = tape.gradient(loss_critic, critic.trainable_variables)
        optimizer_critic.apply_gradients(zip(grads, critic.trainable_variables))
        return tf.squeeze(loss_critic)

@tf.function
def critic_grad_tf(critic, experiences):
    with tf.GradientTape() as tape:
        unstacked_exp = tf.unstack(tf.convert_to_tensor(experiences), axis=1)
        to_stack = []
        actions_wathed_index = []
        for index in range(0,experiences.shape[-1]-3,2): # I consider from first outcome to last one (but guess)
            actions_wathed_index.append(index)
            to_stack.append(tf.reshape(unstacked_exp[index],(experiences.shape[0],1,1)))

        actions_indexed = tf.concat(to_stack,axis=1)
        tape.watch(actions_indexed)

        index_actions=0
        watched_exps=[tf.ones((experiences.shape[0],1,1))*critic.pad_value]
        watched_actions_unstacked = tf.unstack(actions_indexed, axis=1)
        for index in range(0,experiences.shape[-1]-1):
            if index in actions_wathed_index:
                watched_exps.append(tf.expand_dims(watched_actions_unstacked[index_actions], axis=2))
                index_actions+=1
            else:
                watched_exps.append(tf.reshape(unstacked_exp[index],(experiences.shape[0],1,1)))

        qvals = critic(tf.reshape(tf.concat(watched_exps, axis=2), (experiences.shape[0],critic.dolinar_layers+1,2)))

        dq_da = tape.gradient(qvals, actions_indexed)
        return dq_da

@tf.function
def actor_grad_tf(actor, dq_da, experiences, optimizer_actor):
    actor.lstm.stateful=False
    with tf.GradientTape() as tape:
        tape.watch(actor.trainable_variables)
        finns = [tf.ones((experiences.shape[0], 1,1))*actor.pad_value]
        unstacked_exp = tf.unstack(experiences, axis=1)
        for index in range(1,2*actor.dolinar_layers-2,2):
            finns.append(tf.reshape(unstacked_exp[index], (experiences.shape[0], 1,1)))
        final_preds = tf.concat(finns, axis=1)
        final_preds = actor(final_preds)
        da_dtheta=tape.gradient(final_preds, actor.trainable_variables, output_gradients=-dq_da)
        optimizer_actor.apply_gradients(zip(da_dtheta, actor.trainable_variables))
    actor.lstm.stateful=True

    return


@tf.function
def optimization_step(experiences, critic, critic_target, actor, actor_target, optimizer_critic, optimizer_actor):
    # actor.lstm.reset_states()
    # experiences = experiences.astype(np.float32)
    targeted_experience = actor_target.process_sequence_of_experiences_tf(experiences)
    sequences, zeroed_rews = critic_target.process_sequence_tf(targeted_experience)
    labels_critic = critic_target.give_td_errors_tf( sequences, zeroed_rews)

    loss_critic = step_critic_tf(sequences ,labels_critic, critic, optimizer_critic)

    dq_da = critic_grad_tf(critic, experiences)
    actor_grad_tf(actor, dq_da, experiences, optimizer_actor)
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
    print("starting time: {}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    print("saving results in " + str(directory))
    avg_train = []
    ##### STORING FOLDER ####
    ##### STORING FOLDER ####
    ##### STORING FOLDER ####

    at = make_attenuations(dolinar_layers)
    for episode in tqdm(range(total_episodes)):

        env.pick_phase()
        experiences=[] #where the current history of the current episode is stored
        context_outcome_actor = np.reshape(np.array([actor.pad_value]),(1,1,1)).astype(np.float32)
        outcomes_so_far = []
        for layer in range(actor.dolinar_layers):
            beta_would_do = np.squeeze(actor(context_outcome_actor))
            beta =  beta_would_do + np.random.uniform(-noise_displacement, noise_displacement)#np.clip(,-2*amplitude,2*amplitude)
            policy_evaluator.recorded_trajectory_tree[str(layer)][str(np.array(outcomes_so_far))].append(beta)
            policy_evaluator.recorded_trajectory_tree_would_do[str(layer)][str(np.array(outcomes_so_far))].append(beta_would_do)

            outcome = env.give_outcome(beta,layer)
            outcomes_so_far.append(int(outcome))
            experiences.append(beta)
            experiences.append(outcome)
            context_outcome_actor = np.reshape(np.array([outcome]),(1,1,1)).astype(np.float32)

        ### ep-gredy guessing of the phase###
        ### ep-gredy guessing of the phase###
        if np.random.random()<ep_guess:
            val = np.random.choice(range(number_phases),1)[0]
            guess_index, guess_input_network = val, val/critic.number_phases
            # print(guess_input_network)
        else:
            guess_index, guess_input_network = critic.give_favourite_guess(experiences) #experiences is the branch of the current tree of actions + outcomes.
        experiences.append(guess_input_network)

        reward = env.give_reward(guess_index)
        experiences.append(reward)
        buffer.add(tuple(experiences))


        rt.append(reward)
        pt.append(policy_evaluator.greedy_strategy(actor = actor, critic = critic))

        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######
        # if (buffer.count>1):#(episode%100==1):
            # sampled_experiences = buffer.sample(batch_size)
            # np.save(str(dolinar_layers)+"_sample", sampled_experiences)
        if (buffer.count>batch_size):#(episode%100==1):
            sampled_experiences = tf.convert_to_tensor(buffer.sample(batch_size), dtype=np.float32)

            new_loss = optimization_step(sampled_experiences, critic, critic_target, actor, actor_target, optimizer_critic, optimizer_actor)
            new_loss = new_loss.numpy()
            critic_target.update_target_parameters(critic)
            actor_target.update_target_parameters(actor)
            # noise_displacement = max(0.1,0.999*noise_displacement)
        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######
        avg_train.append(new_loss)
        actor.lstm.reset_states()

         #set again the states to zero, because when actor.lstm.stateful = True, it does not reset state along differnt batches !

        # if episode%(total_episodes/100) == 0: #this is for showing 10 results in total.
        #
        #     template = 'Episode {}, \Rt: {}, \Pt: {}, Train loss: {}\n\n'
        #     print(template.format(episode+1,
        #                         np.sum(rt)/(episode+1),
        #                           pt[-1],
        #                          np.round(np.array(avg_train).mean(),5),
        #                         )
        #           )

    cumre=0
    rrt = []
    for k in rt:
        cumre+=k
        rrt.append(cumre)
    rrt = rrt/np.arange(1,len(rt)+1)

    np.save(directory+"/learning_curves/", rrt)
    np.save(directory+"/learning_curves/", pt)
    policy_evaluator.save_hisory_tree(directory+"/action_trees")

    for model, net_folder in zip([actor, actor_target, critic, critic_target],["actor_primary", "actor_target", "critic_primary", "critic_target"]):
        model.save_weights(directory+"/networks/"+net_folder+"/")
    just_plot(rrt, pt, avg_train, env.helstrom(), policy_evaluator, directory)
    # BigPlot(buffer,rt, pt, history_betas, history_betas_would_have_done, histo_preds, losses, directory)
    return


if __name__ == "__main__":
    info_run = ""
    to_csv=[]
    amplitude=0.4
    tau = .01
    lr_critic = 0.0001
    lr_actor=0.001
    noise_displacement = .1
    ep_guess=0.01
    dolinar_layers=2
    number_phases=2

    for buffer_size in [5000.]:

        for batch_size in [8.]:

            name_run = RDPG(amplitude=amplitude, total_episodes=10**2, dolinar_layers=dolinar_layers, noise_displacement=noise_displacement, tau=tau,
        buffer_size=buffer_size, batch_size=batch_size, lr_critic=lr_critic, lr_actor=lr_actor, ep_guess=ep_guess)

    info_run +="***\n***\nname_run: {} ***\ntau: {}\nlr_critic: {}\nnoise_displacement: {}\nbatch_size: {}\n-------\n-------\n\n".format(name_run,tau, lr_critic, noise_displacement, batch_size)

    to_csv.append({"name_run":"run_"+str(name_run), "tau": tau, "lr_critic":lr_critic, "noise_displacement": noise_displacement,
    "BS":batch_size})

    with open("results/info_runs.txt", 'a+') as f:
        f.write(info_run)
        f.close()
    ##### if we put more runs... ###
    # pp = pd.DataFrame(to_csv)
    # pp.to_csv("results/panda_info.csv")


##################time 10**4  without tf.function ################
###with tf.function 22 min 10**4, batch_size =8
#### without batch 16- 1.10 hours, 32. takes ~2 hourse, 64 ~ 3hourse, 128>5 hourse
