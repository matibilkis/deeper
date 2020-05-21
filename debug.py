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
import timeit

amplitude=0.4
dolinar_layers=2
number_phases=2
total_episodes = 10**3
buffer_size=500
batch_size=64
ep_guess=0.01
noise_displacement=0.5
lr_actor=0.01
lr_critic=0.001
tau=0.005


exper = np.load("example_buffer/2_sample.npy")
env = Environment(amplitude=amplitude, dolinar_layers = dolinar_layers, number_phases=number_phases)
# buffer = ReplayBuffer(buffer_size=buffer_size)

critic = Critic(nature="primary",valreg=0.01, dolinar_layers = dolinar_layers, number_phases=number_phases)
critic_target = Critic(nature="target", dolinar_layers = dolinar_layers, number_phases=number_phases)
actor = Actor(nature="primary", dolinar_layers = dolinar_layers)
actor_target = Actor(nature="target", dolinar_layers = dolinar_layers)

optimizer_critic = tf.keras.optimizers.Adam(lr=lr_critic)
optimizer_actor = tf.keras.optimizers.Adam(lr=lr_actor)

policy_evaluator = PolicyEvaluator(amplitude = amplitude, dolinar_layers=dolinar_layers, number_phases = number_phases)

#
experiences = exper.astype(np.float32)
targeted_experience = actor_target.process_sequence_of_experiences_tf(experiences)
sequences, zeroed_rews = critic_target.process_sequence_tf(targeted_experience)
critic_target.give_td_error_Kennedy_guess_tf( sequences.numpy(), zeroed_rews.numpy())
