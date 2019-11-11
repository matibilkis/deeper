import numpy as np
import basics
import misc
import tensorflow as tf
from tensorflow.keras.layers import Dense
import random
from give_actions_kennedy import Give_Action
import os
from memory import Memory
from networks_kennedy import QN_l1, QN_guess_kennedy
from learn_kennedy import learn
import give_probability_kennedy
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 35})


def main(states_wasted=10**4):
    cumulative = []
    success_prob_evolution = []
    cum_rews=0
    alpha = .56
    for episode in range(states_wasted):
        # if episode%100 == 1:
        #     print(episode, " of ", states_wasted)
        #     print("cumulative: ", str(cumulative[-1]/episode))
        #     print("p_s_greedy: ", str(give_probability_kennedy.probability_greedy( alpha, networks= [qn_l1_prim, qn_guess_prim]    )))
        # epsilon = max(0.01, np.exp(-episode/500))
        epsilon = 1
        phase = np.random.choice([-1,1],1)[0]
        labelbeta1, beta1 = qn_l1_prim.give_first_beta(epsilon)
        p0 = np.exp(-(beta1-(phase*np.cos(ats[0])*alpha))**2)
        outcome1 = np.random.choice([0,1],1,p=[p0,1-p0])[0]
        new_state = [outcome1, beta1]
        # labelbeta2, beta2 = qn_l2_prim.give_second_beta(new_state,epsilon)
        # p1 = np.exp(-(beta2-(phase*np.sin(ats[0])*alpha))**2)
        # outcome2 = np.random.choice([0,1],1,p=[p1,1-p1])[0]
        # new_state = [outcome1, outcome2, beta1, beta2]
        label_guess, guess = qn_guess_prim.give_guess(new_state,epsilon)
        if guess == phase:
            reward = 1
        else:
            reward = 0
        buffer.add_sample((outcome1, beta1, labelbeta1, guess, label_guess, reward))
        learn(networks, optimizers, buffer, batch_length=1000, TAU =TAU, episode_info=episode)
        if (episode %100 == 1)&(episode>1):
            print("GUARDANDO!!")
            os.chdir("networks_checkpoints")
            if not os.path.exists(str(episode)):
                os.makedirs(str(episode))
            os.chdir("..")
            os.chdir("q-values")
            if not os.path.exists(str(episode)):
                os.makedirs(str(episode))
            os.chdir("..")
            for i in qn_l1_prim.trainable_variables:
                with open("networks_checkpoints/"+ str(episode) + "/" + str(i.name).replace("/",""), "w") as f:
                    f.write(str(i.numpy()))
                    f.close()
                np.save("networks_checkpoints"+"/"+str(episode), i.numpy())
            q_layer1 = qn_l1_prim(np.expand_dims(np.array([]), axis=0))
            q_guess_0_beta1 = qn_guess_prim(np.expand_dims(beta1_out0, axis=0))
            q_guess_1_beta1 = qn_guess_prim(np.expand_dims(beta1_out1, axis=0))

            np.save("q-values/"+str(episode)+"/qguess0", q_guess_0_beta1)
            np.save("q-values/"+str(episode)+"/qguess1", q_guess_0_beta1)
            np.save("q-values/"+str(episode)+"/qlayer1", q_layer1)

            with open("q-values/"+ str(episode) + "/" + "first_layer_outcome0","w") as f:
                f.write(str(q_guess_0_beta1.numpy()))
                f.close()

            with open("q-values/"+ str(episode) + "/" + "first_layer_outcome1","w") as f:
                f.write(str(q_guess_1_beta1.numpy()))
                f.close()

            with open("q-values/"+ str(episode) + "/" + "fierst_layer","w") as f:
                f.write(str(q_layer1.numpy()))
                f.close()

        cum_rews += reward
        success_prob_evolution.append(give_probability_kennedy.probability_greedy(alpha, networks = [qn_l1_prim, qn_guess_prim]))
        cumulative.append(cum_rews)
        #if episode%10**3 == 0:
         #   save_parameters()
    times = np.arange(1, states_wasted+1)
    np.save("resuts.npy", [times, cumulative, success_prob_evolution], allow_pickle=True)
    with open("time.txt","w") as f:
        f.write(str(datetime.now() - begin))
        f.close()

    with open("hyperparameters.txt","w") as f:
        f.write("lr: "+ str(lr) + "\nTAU: " +str(TAU))
        f.close()

    plt.figure(figsize=(20,20))
    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0))

    ax1.plot(times,cumulative/times, alpha=0.75, color="red")
    ax1.set_ylabel("Freq succ")
    ax2.plot(times,success_prob_evolution, alpha=0.75, color="blue")
    ax2.plot([min(times), max(times)], [opt_prob]*2, alpha=0.5, color="black") #defined when basics.
    ax2.set_ylabel("Prob succ greedy")

    plt.savefig("results_plot.png")
    plt.close()
    return




# taus = [1E-1, 1E-2, 1E-3, 1E-4]
# lr = [1E-1,0.05, 1E-2, 1E-3]
taus = [1E-1]
lr = [0.1]
params = []
for tau in taus:
    for lrs in lr:
        params.append([lrs, tau])

for run in range(len(params)):

    begin = datetime.now()


    basic = basics.Basics(layers=1,resolution=.1, bound_displacements=1)
    basic.define_actions()
    ats = misc.make_attenuations(layers=1)

    opt_prob = np.max(1-basic.err_kennedy(basic.actions[0]))

    lr, TAU = params[run]

    beta1_out0 = []
    beta1_out1 = []
    for beta1 in basic.actions[0]:
        beta1_out0.append([0,beta1])
        beta1_out1.append([1,beta1])

    qn_l1_prim = QN_l1(basic.actions[0], dirname_backup_weights="qn_l1_prim")
    qn_l1_targ = QN_l1(basic.actions[0], dirname_backup_weights="qn_l1_targ")

    qn_guess_prim = QN_guess_kennedy(basic.possible_phases, dirname_backup_weights="qn_guess_prim")
    qn_guess_targ = QN_guess_kennedy(basic.possible_phases, dirname_backup_weights="qn_guess_prim")

    networks = [qn_l1_prim, qn_l1_targ, qn_guess_prim, qn_guess_targ]

    optimizer_l1 = tf.keras.optimizers.Adam(lr=lr)

    optimizer_guess = tf.keras.optimizers.Adam(lr=lr)
    optimizers = [optimizer_l1, optimizer_guess]

    buffer = Memory(10**6,load_path=None)

    r = misc.Record("Results-epsilon1")
    main(10**3 + 1)
    os.chdir("../..")
    # buffer.export()

    ####
# os.system("python3 todo.py")
