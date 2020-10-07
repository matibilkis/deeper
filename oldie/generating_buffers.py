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




def RDPG(special_name="", amplitude=0.4, dolinar_layers=2, number_phases=2, total_episodes = 10**3, buffer_size=500, batch_size=64, ep_guess=0.01, noise_displacement=0.5, lr_actor=0.01, lr_critic=0.001, tau=0.005):

    env = Environment(amplitude=amplitude, dolinar_layers = dolinar_layers, number_phases=number_phases)
    buffer = ReplayBuffer(buffer_size=buffer_size)

    actor = Actor(nature="primary", dolinar_layers = dolinar_layers)


    for episode in tqdm(range(total_episodes)):

        env.pick_phase()
        experiences=[] #where the current history of the current episode is stored
        context_outcome_actor = np.reshape(np.array([actor.pad_value]),(1,1,1)).astype(np.float32)
        outcomes_so_far = []
        for layer in range(actor.dolinar_layers):
            # beta_would_do = np.squeeze(actor(context_outcome_actor))
            # beta =  beta_would_do + np.random.uniform(-noise_displacement, noise_displacement)#np.clip(,-2*amplitude,2*amplitude)
            beta = np.random.choice(np.arange(-1.0,1.0,.1))
            outcome = env.give_outcome(beta,layer)
            outcomes_so_far.append(int(outcome))
            experiences.append(beta)
            experiences.append(outcome)
            context_outcome_actor = np.reshape(np.array([outcome]),(1,1,1)).astype(np.float32)


        guess_index = np.random.choice(range(number_phases),1)[0]
        experiences.append(guess_index)
        reward = env.give_reward(guess_index)#, modality="pt",history = experiences[:-1])
        experiences.append(reward)
        buffer.add(tuple(experiences))

    np.save("buffers/2L_r",buffer.sample(buffer.buffer_size) )
    return


if __name__ == "__main__":
    info_run = ""
    to_csv=[]
    amplitude=0.4
    tau = 0.5*10**-4
    lr_critic = 10**-4
    lr_actor=10**-4
    noise_displacement = 1.
    ep_guess=0.01
    dolinar_layers=2
    number_phases=2
    buffer_size = 10**6.
    #no_delete_variables =
    #["no_delete_variables","amplitude", "to_csv","tau", "lr_critic", "lr_actor", "noise_displacement", "ep_guess", "dolinar_layers", "number_phases", "buffer_size", "batch_size"]

    for batch_size in [8.]:

        name_run = RDPG(amplitude=amplitude, total_episodes=10**5, dolinar_layers=dolinar_layers, noise_displacement=noise_displacement, tau=tau, buffer_size=buffer_size, batch_size=batch_size, lr_critic=lr_critic, lr_actor=lr_actor, ep_guess=ep_guess)
