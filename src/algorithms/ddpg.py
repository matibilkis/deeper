"""
Deep Deterministic Policy Gradient (DDPG) algorithm implementation.

Main training loop and optimization steps for DDPG applied to quantum receiver optimization.
"""

import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

try:
    from src.networks.actor import Actor
    from src.networks.critic import Critic
    from src.algorithms.buffer import ReplayBuffer
    from src.quantum.receivers import Prob, qval, ps_maxlik
    from src.utils.misc import record
    from src.utils.plots import plot_learning_curves, plot_beta_histogram
except ImportError:
    # For relative imports when running as module
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.networks.actor import Actor
    from src.networks.critic import Critic
    from src.algorithms.buffer import ReplayBuffer
    from src.quantum.receivers import Prob, qval, ps_maxlik
    from src.utils.misc import record
    from src.utils.plots import plot_learning_curves, plot_beta_histogram


def optimization_step(experiences, critic, critic_target, actor, optimizer_critic, optimizer_actor, train_loss):
    """
    Perform one optimization step for DDPG.
    
    Updates both critic and actor networks using the sampled experiences.
    
    Args:
        experiences: Batch of experiences from replay buffer
        critic: Primary critic network
        critic_target: Target critic network
        actor: Actor network
        optimizer_critic: Optimizer for critic network
        optimizer_actor: Optimizer for actor network
        train_loss: TensorFlow metric for tracking training loss
    """
    sequences, zeroed_rews = critic.process_sequence(experiences)
    labels_critic = critic_target.give_td_error_Kennedy_guess(sequences, zeroed_rews)
    
    # Update critic
    with tf.GradientTape() as tape:
        tape.watch(critic.trainable_variables)
        preds_critic = critic(sequences)
        loss_critic = tf.keras.losses.MSE(labels_critic, preds_critic)
        loss_critic = tf.reduce_mean(loss_critic)
        grads = tape.gradient(loss_critic, critic.trainable_variables)
        optimizer_critic.apply_gradients(zip(grads, critic.trainable_variables))
        train_loss(loss_critic)

    # Update actor using policy gradient
    with tf.GradientTape() as tape:
        ones = tf.ones(shape=(experiences.shape[0], 1)) * critic.pad_value
        actions = actor(np.expand_dims(np.zeros(len(experiences)), axis=1))
        tape.watch(actions)
        qvals = critic(tf.expand_dims(tf.concat([actions, ones], axis=1), axis=1))
        dq_da = tape.gradient(qvals, actions)

    with tf.GradientTape() as tape:
        actionss = actor(np.expand_dims(np.zeros(len(experiences)), axis=1))
        da_dtheta = tape.gradient(actionss, actor.trainable_variables, output_gradients=-dq_da)

    optimizer_actor.apply_gradients(zip(da_dtheta, actor.trainable_variables))
    return


def ddpg_kennedy(special_name="", total_episodes=10**3, buffer_size=500, batch_size=64, 
                 ep_guess=0.1, noise_displacement=0.5, lr_actor=0.01, lr_critic=0.001, 
                 tau=0.005, repetitions=1, plots=True):
    """
    Main DDPG training function for Kennedy receiver optimization.
    
    Args:
        special_name: Custom name for this run (default: auto-generated)
        total_episodes: Total number of training episodes
        buffer_size: Size of experience replay buffer
        batch_size: Batch size for training
        ep_guess: Epsilon for epsilon-greedy guess selection
        noise_displacement: Noise added to actions for exploration
        lr_actor: Learning rate for actor network
        lr_critic: Learning rate for critic network
        tau: Soft update coefficient for target networks
        repetitions: Number of optimization steps per episode
        plots: Whether to generate plots (default: True)
        
    Returns:
        Directory path where results are saved
    """
    if not os.path.exists("results"):
        os.makedirs("results")

    amplitude = 0.4
    buffer = ReplayBuffer(buffer_size=buffer_size)

    critic = Critic(valreg=0.01)
    critic_target = Critic()
    actor = Actor(input_dim=1)

    # Initialize network
    actor(np.array([[0.]]).astype(np.float32))
    
    optimizer_critic = tf.keras.optimizers.Adam(lr=lr_critic)
    optimizer_actor = tf.keras.optimizers.Adam(lr=lr_actor)

    rt = []
    pt = []

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    if special_name == "":
        numb = record()
        current_run_and_time = "results/run_" + str(numb)
    else:
        current_run_and_time = "results/" + special_name

    directory = current_run_and_time
    os.makedirs(directory, exist_ok=True)
    
    train_log = current_run_and_time + '/train_l0'
    test_log = current_run_and_time + '/test_l0'

    train_summary_writer = tf.summary.create_file_writer(train_log)
    test_summary_writer_0 = tf.summary.create_file_writer(test_log)

    info_optimizers = "optimizer_critic: {} \nOptimizer_actor: {}\n".format(
        optimizer_critic.get_config(), optimizer_actor.get_config()
    )
    info_buffer = "Buffer_size: {}\n Batch_size for sampling: {}\n".format(
        buffer.buffer_size, batch_size
    )
    info_epsilons = "epsilon-guess: {}\nepsilon_displacement_noise: {}".format(
        ep_guess, noise_displacement
    )

    data = ("tau: {}, repetitions per optimization step: {}\n\n"
            "**** optimizers ***\n" + info_optimizers + "\n\n"
            "*** BUFFER ***\n" + info_buffer + "\n\n"
            " *** NOISE PARAMETERS *** \n" + info_epsilons)
    
    with open(directory + "/info.txt", 'w') as f:
        f.write(data)
        f.close()

    print("Beginning to train!\n\n")
    print(data)
    print("starting time: {}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    print("saving results in " + str(directory))
    
    avg_train = []
    avg_test = []
    history_betas = []
    history_betas_would_have_done = []
    histo_preds = {"layer0": {}, "layer1": {}}

    for episode in tqdm(range(total_episodes)):
        alice_phase = np.random.choice([-1., 1.], 1)[0]
        beta_would_do = actor(np.array([[0.]])).numpy()[0][0]
        beta = beta_would_do + np.random.uniform(-noise_displacement, noise_displacement)
        proboutcome = Prob(alice_phase * amplitude, beta, 0.)
        outcome = np.random.choice([0., 1.], 1, p=[proboutcome, 1 - proboutcome])[0]

        history_betas.append(beta)
        history_betas_would_have_done.append(beta_would_do)

        # Epsilon-greedy guess selection
        if np.random.random() < ep_guess:
            guess = np.random.choice([-1., 1.], 1)[0]
        else:
            guess = critic.give_favourite_guess(critic.pad_single_sequence([beta, outcome, 1.]))

        if guess == alice_phase:
            reward = 1.
        else:
            reward = 0.

        buffer.add(beta, outcome, guess, reward)

        # Optimization step
        if buffer.count > 1:
            experiences = buffer.sample(batch_size)
            optimization_step(experiences, critic, critic_target, actor, 
                          optimizer_critic, optimizer_actor, train_loss)
            critic_target.update_target_parameters(critic, tau=tau)

        avg_train.append(train_loss.result().numpy())
        avg_test.append(0.)

        # Calculate success probability if agent went greedy
        p = 0
        for outcome_val in [0., 1.]:
            guess_val = critic.give_favourite_guess(
                critic.pad_single_sequence([beta_would_do, outcome_val, 1.])
            )
            p += Prob(guess_val * amplitude, beta_would_do, outcome_val)
        p /= 2
        pt.append(p)

        rt.append(reward)

        # Periodic logging and plotting
        if episode % (total_episodes // 10) == 0:
            template = ('Episode {}, Rt: {}, Pt: {}, Train loss: {}, Test loss: {}\n\n')
            print(template.format(
                episode + 1,
                np.sum(rt) / (episode + 1),
                pt[-1],
                np.round(train_loss.result().numpy(), 5),
                np.round(test_loss.result().numpy(), 5)
            ))

    # Calculate cumulative rewards
    rt = [np.sum(rt[:k]) for k in range(len(rt))]
    rt = rt / np.arange(1, len(rt) + 1)

    losses = [avg_train, avg_test]

    # Generate plots if requested
    if plots:
        optimal = max([ps_maxlik(b) for b in buffer.betas])
        plot_learning_curves(rt, pt, optimal, directory, losses)
        plot_beta_histogram(history_betas, history_betas_would_have_done, buffer.betas, directory)

    return directory

