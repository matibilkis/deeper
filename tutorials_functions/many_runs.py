
from nets import Critic, Actor
import tensorflow as tf
import numpy as np
from misc import *

for k in range(2):
    optimizer_critic = tf.keras.optimizers.Adam(lr=0.1)
    optimizer_actor = tf.keras.optimizers.Adam(lr=0.1)

    experiences = np.load("buffers/1L-stoch.npy").astype(np.float32)[:10]
    critic = Critic(nature="primary", dolinar_layers=1)
    actor = Actor(nature="primary", dolinar_layers=1)
    critic_target = Critic(nature="target", dolinar_layers=1)
    actor_target = Actor(nature="target", dolinar_layers=1)

    targeted_experience = actor_target.process_sequence_of_experiences_tf(experiences)
    batched_input, zeroed_rews = critic_target.process_sequence_tf(targeted_experience)
    labels_critic = critic_target.give_td_errors_tf( batched_input, zeroed_rews)

    loss_critic = critic.step_critic_tf(batched_input ,labels_critic, optimizer_critic)

    dq_da = critic.critic_grad_tf(experiences)
    actor.actor_grad_tf(dq_da, experiences, optimizer_actor)


    print(k)
