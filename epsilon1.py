import numpy as np
import basics
import misc
import tensorflow as tf
from tensorflow.keras.layers import Dense
import random
from give_actions import Give_Action
import os
from memory import Memory
from networks import QN_l1, QN_l2, QN_guess
from learn import learn
import give_probability
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 35})


def main(states_wasted=10**4):
    cumulative = []
    success_prob_evolution = []
    cum_rews=0
    alpha = .56
    for episode in range(states_wasted):
        if episode%100 == 0:
            print(episode, " of ", states_wasted)
            if episode%100==1:
                print("cumulative: ", str(cumulative[-1]/episode))
                print("p_s_greedy: ", str(give_probability.probability_greedy( ats, alpha, networks = [qn_l1_prim, qn_l2_prim, qn_guess_prim]    )))
        # epsilon = max(0.01, np.exp(-episode/500))
        epsilon = 1
        phase = np.random.choice([-1,1],1)[0]
        labelbeta1, beta1 = qn_l1_prim.give_first_beta(epsilon)
        p0 = np.exp(-(beta1-(phase*np.cos(ats[0])*alpha))**2)
        outcome1 = np.random.choice([0,1],1,p=[p0,1-p0])[0]
        new_state = [outcome1, beta1]
        labelbeta2, beta2 = qn_l2_prim.give_second_beta(new_state,epsilon)
        p1 = np.exp(-(beta2-(phase*np.sin(ats[0])*alpha))**2)
        outcome2 = np.random.choice([0,1],1,p=[p1,1-p1])[0]
        new_state = [outcome1, outcome2, beta1, beta2]
        label_guess, guess = qn_guess_prim.give_guess(new_state,epsilon)
        if guess == phase:
            reward = 1
        else:
            reward = 0
        buffer.add_sample((outcome1, outcome2, beta1, beta2, labelbeta1, labelbeta2, guess, label_guess, reward))
        if episode > 1:
            learn(networks, optimizers, buffer, batch_length=episode, TAU =TAU)

            # if episode % 10 == 0:
            #     for i in networks:
            #         i.save_now()
        cum_rews += reward
        success_prob_evolution.append(give_probability.probability_greedy(ats, alpha, networks = [qn_l1_prim, qn_l2_prim, qn_guess_prim]))
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
    ax2.plot([min(times), max(times)], [1-0.09779934348580333
]*2, alpha=0.5, color="black")
    ax2.set_ylabel("Prob succ greedy")

    plt.savefig("results_plot.png")
    plt.close()
    return




taus = [1E-1, 1E-2, 1E-3, 1E-4]
lr = [1E-1,0.05, 1E-2, 1E-3]
params = []
for tau in taus:
    for lrs in lr:
        params.append([lrs, tau])

for run in range(5):

    begin = datetime.now()


    basic = basics.Basics(resolution=.1, bound_displacements=1)
    basic.define_actions()
    ats = misc.make_attenuations(layers=2)

    lr, TAU = params[run]




    qn_l1_prim = QN_l1(basic.actions[0], dirname_backup_weights="qn_l1_prim")
    qn_l1_targ = QN_l1(basic.actions[0], dirname_backup_weights="qn_l1_targ")
    qn_l2_prim = QN_l2(basic.actions[1], dirname_backup_weights="qn_l2_prim")
    qn_l2_targ = QN_l2(basic.actions[1], dirname_backup_weights="qn_l2_targ")
    qn_guess_prim = QN_guess(basic.possible_phases, dirname_backup_weights="qn_guess_prim")
    qn_guess_targ = QN_guess(basic.possible_phases, dirname_backup_weights="qn_guess_prim")

    networks = [qn_l1_prim, qn_l1_targ, qn_l2_prim, qn_l2_targ, qn_guess_prim, qn_guess_targ]

    optimizer_l1 = tf.keras.optimizers.Adam(lr=lr)
    optimizer_l2 = tf.keras.optimizers.Adam(lr=lr)
    optimizer_guess = tf.keras.optimizers.Adam(lr=lr)
    optimizers = [optimizer_l1, optimizer_l2, optimizer_guess]

    buffer = Memory(10E4)

    r = misc.Record("Results-epsilon1")
    main(10**4)
    os.chdir("../..")

    ####
os.system("python3 todo.py")
