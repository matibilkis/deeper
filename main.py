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
from plots import just_plot, profiles_kennedy
import misc
import nets
from buffer import ReplayBuffer
from datetime import datetime
from art import text2art
# tf.config.experimental_run_functions_eagerly(True)

##### TO DO: see what's wrong with internal states assigment of the lstm when
# I change the batch_size.

#On way to overcome this is to pad all values, so you don't need to reset batch size.

giq4ever = text2art("%GIQ \n4ever")

def RDPG(special_name="", amplitude=0.4, dolinar_layers=2, number_phases=2, total_episodes = 10**3, buffer_size=500, batch_size=64, ep_guess=0.01,
 noise_displacement=0.5, lr_actor=0.01, lr_critic=0.001, tau=0.005, reduce_noise=True, min_noise_value=0.05):

    if not os.path.exists("results"):
        os.makedirs("results")

    env = Environment(amplitude=amplitude, dolinar_layers = dolinar_layers, number_phases=number_phases)
    buffer = ReplayBuffer(buffer_size=buffer_size)

    critic = nets.Critic(nature="primary", dolinar_layers = dolinar_layers, number_phases=number_phases)
    critic_target = nets.Critic(nature="target", dolinar_layers = dolinar_layers, number_phases=number_phases, tau = tau)
    actor = nets.Actor(nature="primary", dolinar_layers = dolinar_layers, batch_size_info=batch_size)
    actor_target = nets.Actor(nature="target", dolinar_layers = dolinar_layers, tau = tau, batch_size_info=batch_size)

    optimizer_critic = tf.keras.optimizers.Adam(lr=lr_critic)
    optimizer_actor = tf.keras.optimizers.Adam(lr=lr_actor)

    policy_evaluator = misc.PolicyEvaluator(amplitude = amplitude, dolinar_layers=dolinar_layers, number_phases = number_phases)

    rt = []
    pt = []
    pt_max_like=[]
    new_loss=0.

    ##### STORING FOLDER ####
    ##### STORING FOLDER ####
    ##### STORING FOLDER ####
    numb = misc.record()
    directory ="results/run_" + str(numb)

    info = giq4ever
    info += "\n\n -------------- Optimizers info: -------------- \nOptimizer_critic: {} \nOptimizer_actor: {}\n".format(optimizer_critic.get_config(), optimizer_actor.get_config())

    info += "\n\n --------------  Buffer --------------\nBuffer Size: {}\n Batch_size for sampling: {}\n".format(buffer.buffer_size, batch_size)
    info += "\n\n ------- Noise info ------- \nepsilon-guess: {}\n Initial noise of displaacements: {} \n Reducing noise of displacements? {} \n Min_noise_value: {}".format(ep_guess,noise_displacement, reduce_noise,min_noise_value)

    info+= "\n\n --------------More hyperparameters and CV info-------------- \n\ntau: {} \nAmplitude: {}\nDolinar Layers: {}\nNumber of phases: {}".format(tau, env.amplitude, actor.dolinar_layers, env.number_phases)
    info+="\n\n********************\n\n********************\n\n********************"
    info+=text2art("That's all folks!")

    with open(directory+"/info.txt", 'w') as f:
        f.write(info)
        f.close()

    print(text2art("Beggining the train! "))
    print("\n\n")
    print("Starting time: {}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    print("\n\n")
    print("saving results in " + str(directory))
    ##### STORING FOLDER ####
    ##### STORING FOLDER ####
    ##### STORING FOLDER ####
    avg_train = []
    my_tau = (total_episodes)/np.log(1/min_noise_value**4) #for noise reduction, by the half of the experiment you begin to exploit. the max is because i'm sure i'll sometime set it to zero and forget about this. But it's called only one per run, so no worries :)
    #The idea of this is that i get to min_noise_value by half of the total_episodes, so i "explore" half and exploit the other half.
### e^{-t/\tau} = \ep0 -----> \tau = \frac{t}{\log (1\ep0)

### just to initialize the netwroks####
    context_outcome_actor = np.reshape(np.array([actor.pad_value]*batch_size),(batch_size,1,1)).astype(np.float32)
    actor(context_outcome_actor)
    actor_target(context_outcome_actor)
    actor.lstm.reset_states()
### just to initialize the netwroks####

    for episode in tqdm(range(total_episodes)):

        env.pick_phase()
        experiences=[] #where the current history of the current episode is stored
        context_outcome_actor = np.reshape(np.array([actor.pad_value]*batch_size),(batch_size,1,1)).astype(np.float32) #this is fixed because actor has stateful lstm. Changing the atch_size didn't worked to me if excution was not eagerly.
        outcomes_so_far = []
        for layer in range(actor.dolinar_layers):
            beta_would_do = np.squeeze(actor(context_outcome_actor)[0]) #don't worry, actor is deerrmnistic hence all batch_size elemtens are same
            beta =  beta_would_do + np.random.uniform(-noise_displacement, noise_displacement)#np.clip(,-2*amplitude,2*amplitude)
            policy_evaluator.recorded_trajectory_tree[str(layer)][str(np.array(outcomes_so_far))].append(beta)
            policy_evaluator.recorded_trajectory_tree_would_do[str(layer)][str(np.array(outcomes_so_far))].append(beta_would_do)

            outcome = env.give_outcome(beta,layer)
            outcomes_so_far.append(int(outcome))
            experiences.append(beta)
            experiences.append(outcome)
            context_outcome_actor = np.reshape(np.array([outcome]*batch_size),(batch_size,1,1)).astype(np.float32)


        ### ep-gredy guessing of the phase###
        # ### ep-gredy guessing of the phase###
        if np.random.random()<ep_guess:
            guess_index = np.random.choice(range(number_phases),1)[0]
        else:
            guess_index = critic.give_favourite_guess(experiences) #experiences is the branch of the current tree

        experiences.append(guess_index)
        reward = env.give_reward(guess_index, modality="bit_stochastic", history = experiences[:-1])
        experiences.append(reward)

        buffer.add(tuple(experiences))

        rt.append(reward)
        #notice it's important states of lstm actor are reset before calling this guy.
        #update: this is done inside misc.optimization_step
        pt.append(policy_evaluator.greedy_strategy(actor = actor, critic = critic, max_like_guess=False)) #information batch_size encoded in actor.batch_size_info
        pt_max_like.append(policy_evaluator.greedy_strategy(actor = actor, critic = critic, max_like_guess=True)) #information batch_size encoded in actor.batch_size_info

        if (buffer.count>batch_size):
            sampled_experiences = tf.convert_to_tensor(buffer.sample(batch_size), dtype=np.float32)
            new_loss = misc.optimization_step(sampled_experiences, critic, critic_target, actor, actor_target, optimizer_critic, optimizer_actor,batch_size, episode)
            new_loss = new_loss.numpy()


            critic_target.update_target_parameters(critic)
            if actor_target.dolinar_layers>1: #otherwise is unusedp
                actor_target.update_target_parameters(actor)
            if reduce_noise:
                noise_displacement = max(min_noise_value,np.exp(-episode/my_tau))

            if episode%(int((total_episodes-batch_size)/10)) == 1:
                var_exists = 'history_predictions' in locals() or 'history_predictions' in globals()
                if not var_exists:
                    history_predictions={"final_episode_info":total_episodes}

                history_predictions[str(episode)] = {"[]":[], "00":[],"01":[],"11":[],"10":[]}
                bbbs = np.arange(.1,1.1,.05)
                inps = np.stack([np.ones(len(bbbs))*critic.pad_value, bbbs], axis=1)
                inps = np.reshape(inps, (len(bbbs),1,2))
                history_predictions[str(episode)]["[]"] = np.squeeze(critic(inps))
                for outcome in [0.,1.]:
                   for guess_index in [0.,1.]:
                        m=[]
                        for k in tf.unstack(inps):
                            m.append(tf.concat([k, np.reshape(np.array([outcome,guess_index]), (1,2))], axis=0))
                        history_predictions[str(episode)][str(outcome)+str(guess_index)] = np.squeeze(critic(tf.stack(m, axis=0)))[:,1]
        ###### OPTIMIZATION STEP ######
        ###### OPTIMIZATION STEP ######

        avg_train.append(new_loss)
        actor.lstm.reset_states()

    cumre=0
    rrt = []
    for k in rt:
        cumre+=k
        rrt.append(cumre)
    rrt = rrt/np.arange(1,len(rt)+1)

    np.save(directory+"/learning_curves/rt", rrt)
    np.save(directory+"/learning_curves/pt_max_like", pt_max_like)
    np.save(directory+"/learning_curves/pt_raw", pt)

    policy_evaluator.save_hisory_tree(directory+"/action_trees")

    for model, net_folder in zip([actor, actor_target, critic, critic_target],["actor_primary", "actor_target", "critic_primary", "critic_target"]):
        model.save_weights(directory+"/networks/"+net_folder+"/")
    just_plot(rrt, pt, avg_train, env.helstrom(), policy_evaluator, directory)
    if dolinar_layers  ==1:
        profiles_kennedy(critic, directory, history_predictions)
    # BigPlot(buffer,rt, pt, history_betas, history_betas_would_have_done, histo_preds, losses, directory)
    return "run_"+str(numb)


info_run = ""
to_csv=[]
amplitude=0.4
lr_critic = 0.01
lr_actor = 0.005
ep_guess=.01
dolinar_layers=1
number_phases=2
tau = 0.01
batch_size=7
noise_displacement=1.
reduce_noise=True



name_run=0
info_runs="::Information on all runs::\n\n"
for batch_size in [8, 16, 64, 128]:
    for buffer_size in [5*10**2, 10**3, 10**5]:
        for ep_greedy in [.01, .3, 1.]:
            for noise_displacement in [.25, .5,1.]:

                begin = datetime.now()
            #     name_run = RDPG(amplitude=amplitude, total_episodes=7*10**4, dolinar_layers=dolinar_layers, noise_displacement=noise_displacement, tau=tau,
            # buffer_size=buffer_size, batch_size=batch_size, lr_critic=lr_critic, lr_actor=lr_actor, ep_guess=ep_greedy, reduce_noise=reduce_noise)

                # infos_run +="***\n***\nname_run: {} ***\n\n\n\n Some details: \n\n tau: {}\nlr_critic: {}\nnoise_displacement: {}\nbatch_size: {}\n-------\n-------\n\n".format(name_run,tau, lr_critic, noise_displacement, batch_size)
                # infos_run += "Noise reduction: {} \nep_guess: {}".format(reduce_noise, ep_guess)
                info_runs+="name_run: {}\ntotal_time: {}\nbuffer_size: {}\nep_greedy: {}\n noise_displacement: {}\nbatch_size: {}\n".format(name_run,str(datetime.now()- begin), buffer_size, ep_greedy, noise_displacement,batch_size)
                name_run+=1



with open("results/info_all_runs.txt", 'w') as f:
    f.write(info_runs)
    f.close()
