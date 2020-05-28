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
        # ### ep-gredy guessing of the phase###
        if np.random.random()<0.3:
            guess_index = np.random.choice(range(number_phases),1)[0]
            # guess_index, guess_phase = val, pol.possible_phases[val]
        else:
            guess_index = critic.give_favourite_guess(experiences) #experiences is the branch of the current tree of actions + outcomes.
        # guess_index, guess_input_network = policy_evaluator.give_max_lik_guess(history = experiences[:-1], return_index = True)

        experiences.append(guess_index)
        reward = env.give_reward(guess_index, modality="probs", history = experiences[:-1])

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

            actor.lstm.stateful=False
            actor.reset_states_workaround(new_batch_size=int(batch_size))

            new_loss = optimization_step(sampled_experiences, critic, critic_target, actor, actor_target, optimizer_critic, optimizer_actor)
            new_loss = new_loss.numpy()
            actor.reset_states_workaround(new_batch_size=1)
            actor.lstm.stateful=True

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
    tau = 0.5*10**-4
    lr_critic = 10**-4
    lr_actor=10**-4
    noise_displacement = .25
    ep_guess=0.01
    dolinar_layers=1
    number_phases=2
    buffer_size = 5000.
    #no_delete_variables =
    #["no_delete_variables","amplitude", "to_csv","tau", "lr_critic", "lr_actor", "noise_displacement", "ep_guess", "dolinar_layers", "number_phases", "buffer_size", "batch_size"]

    for batch_size in [8.]:

        name_run = RDPG(amplitude=amplitude, total_episodes=5*10**2, dolinar_layers=dolinar_layers, noise_displacement=noise_displacement, tau=tau,
    buffer_size=buffer_size, batch_size=batch_size, lr_critic=lr_critic, lr_actor=lr_actor, ep_guess=ep_guess)

        info_run +="***\n***\nname_run: {} ***\ntau: {}\nlr_critic: {}\nnoise_displacement: {}\nbatch_size: {}\n-------\n-------\n\n".format(name_run,tau, lr_critic, noise_displacement, batch_size)

        to_csv.append({"name_run":"run_"+str(name_run), "tau": tau, "lr_critic":lr_critic, "noise_displacement": noise_displacement,
        "BS":batch_size})

        with open("results/info_runs.txt", 'a+') as f:
            f.write(info_run)
            f.close()

        for name in dir():
            if (name.startswith('_'))|(name in ["RDPG", "no_delete_variables","amplitude", "to_csv","tau", "lr_critic", "lr_actor", "noise_displacement", "ep_guess", "dolinar_layers", "number_phases", "buffer_size", "batch_size"]):
                pass
            else:
                del globals()[name]



    ##### if we put more runs... ###
    # pp = pd.DataFrame(to_csv)
    # pp.to_csv("results/panda_info.csv")


##################time 10**4  without tf.function ################
###with tf.function 22 min 10**4, batch_size =8
#### without batch 16- 1.10 hours, 32. takes ~2 hourse, 64 ~ 3hourse, 128>5 hourse
