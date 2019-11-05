import tensorflow as tf
import numpy as np

def learn(networks, optimizers, buffer, batch_length=10E3, TAU=0.01):
    batch = buffer.sample(batch_length)

    qn_l1_prim, qn_l1_targ, qn_l2_prim, qn_l2_targ, qn_guess_prim, qn_guess_targ = networks

    s_2_batch = np.array([[ v[0], v[2]] for v in batch ] )
    labels_beta1 = np.array([v[4] for v in batch])

    q_2_prim = qn_l2_prim(np.expand_dims(s_2_batch, axis=0))
    q_2_prim = np.squeeze(q_2_prim.numpy())

    opt_a_2_prim = np.argmax(q_2_prim,axis=1)

    update_for_q_1_prim = qn_l1_targ(np.expand_dims(np.array([[] for i in range(len(batch))]), axis=0)) #targ = target
    update_for_q_1_prim = np.squeeze(update_for_q_1_prim, axis=0)
    qlabels_l1 = update_for_q_1_prim.copy()
    qlabels_l1[np.arange(batch_length), labels_beta1] = np.squeeze(qn_l2_targ(np.expand_dims(s_2_batch, axis=0)).numpy())[np.arange(batch_length),opt_a_2_prim]

    optimizer_ql1, optimizer_ql2, optimizer_guess = optimizers


    with tf.device("/cpu:0"):
        with tf.GradientTape() as tape:
            tape.watch(qn_l1_prim.trainable_variables)
            pred_q_1s = qn_l1_prim(np.expand_dims(np.array([[] for i in range(len(batch))]), axis=0))

            loss_sum =tf.keras.losses.MSE(pred_q_1s, qlabels_l1)
            loss = tf.reduce_mean(loss_sum)

            grads = tape.gradient(loss, qn_l1_prim.trainable_variables)

            optimizer_ql1.apply_gradients(zip(grads, qn_l1_prim.trainable_variables))

    s_2_batch = np.array([[v[0], v[2]] for v in batch])
    s_3_batch = np.array([[v[0], v[1], v[2], v[3]] for v in batch])

    #labels_guess = np.array([v[7] for v in batch])
    labels_action_2 = np.array([v[5] for v in batch])

    q_3_prim = qn_guess_prim(np.expand_dims(s_3_batch, axis=0))
    q_3_prim = np.squeeze(q_3_prim.numpy())

    opt_a_3_prim = np.argmax(q_3_prim, axis=1)

    update_for_q_2_prim = qn_l2_targ(np.expand_dims(s_2_batch, axis=0))
    update_for_q_2_prim = np.squeeze(update_for_q_2_prim, axis=0)
    qlabels_l2 = update_for_q_2_prim.copy()
    qlabels_l2[np.arange(batch_length), labels_action_2] = np.squeeze(qn_guess_targ(np.expand_dims(s_3_batch, axis=0)).numpy())[np.arange(batch_length), opt_a_3_prim]


    with tf.device("/cpu:0"):
        with tf.GradientTape() as tape:
            tape.watch(qn_l2_prim.trainable_variables)
            pred_q_2s = qn_l2_prim(np.expand_dims(s_2_batch, axis=0))
            loss_sum =tf.keras.losses.MSE(pred_q_2s, qlabels_l2)
            loss = tf.reduce_mean(loss_sum)

            grads = tape.gradient(loss, qn_l2_prim.trainable_variables)
            optimizer_ql2.apply_gradients(zip(grads, qn_l2_prim.trainable_variables))


    s_3_batch = np.array([[v[0], v[1], v[2], v[3]] for v in batch])
    rewards = np.array([v[-1] for v in batch])
    labels_guess = np.array([v[7] for v in batch])

    update_for_q_3_prim = qn_guess_targ(np.expand_dims(s_3_batch, axis=0))
    update_for_q_3_prim = np.squeeze(update_for_q_3_prim, axis=0)
    qlabels_l3 = update_for_q_3_prim.copy()
    qlabels_l3[np.arange(batch_length), labels_guess] = rewards[np.arange(batch_length)]


    with tf.device("/cpu:0"):
        with tf.GradientTape() as tape:
            tape.watch(qn_guess_prim.trainable_variables)
            pred_q_3s = qn_guess_prim(np.expand_dims(s_3_batch, axis=0))
            loss_sum =tf.keras.losses.MSE(pred_q_3s, qlabels_l3)
            loss = tf.reduce_mean(loss_sum)

            grads = tape.gradient(loss, qn_guess_prim.trainable_variables)
            optimizer_guess.apply_gradients(zip(grads, qn_guess_prim.trainable_variables))

    for t, e in zip(qn_l1_targ.trainable_variables, qn_l1_prim.trainable_variables):
        t.assign(t*(1-TAU) + e*TAU)

    for t, e in zip(qn_l2_targ.trainable_variables, qn_l2_prim.trainable_variables):
        t.assign(t*(1-TAU) + e*TAU)

    for t, e in zip(qn_guess_targ.trainable_variables, qn_guess_prim.trainable_variables):
        t.assign(t*(1-TAU) + e*TAU)
    return
