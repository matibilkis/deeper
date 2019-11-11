import tensorflow as tf
import numpy as np
from memory import Memory
from basics import Basics



from networks_kennedy import QN_l1, QN_guess_kennedy
import give_probability_kennedy




basics = Basics(layers=1)
basics.define_actions()
b =np.load("buffer.npy", allow_pickle=True)
buffer = Memory(len(b), load_path=None)
for i in range(len(b)):
    buffer.add_sample(tuple(b[i]))





qn_l1_prim = QN_l1(basics.actions[0])
qn_l1_targ = QN_l1(basics.actions[0])
qn_guess_prim = QN_guess_kennedy(basics.possible_phases)
qn_guess_targ = QN_guess_kennedy(basics.possible_phases)

qvalues1 = qn_l1_prim(np.expand_dims(np.array([]), axis=0))
qvalues1targ = qn_l1_targ(np.expand_dims(np.array([]), axis=0))
qvalueslayerguess_prim = qn_guess_prim(np.expand_dims(np.array([1,-.7]), axis=0))
qvalueslayerguess_targ = qn_guess_targ(np.expand_dims(np.array([1,-.7]), axis=0))



TAU = 0.01
optimizer = tf.keras.optimizers.SGD(lr=0.01)
optimizer_guess = tf.keras.optimizers.SGD(lr=0.01)




def learn_batch(networks , batch_length):
    batch = buffer.sample(batch_length)
    qn_l1_prim, qn_l1_targ, qn_guess_prim, qn_guess_targ = networks

    outcome_1_beta1_batch = np.array([[ int(v[0]), v[1]] for v in batch ] )
    labels_beta1 = np.array([int(v[2]) for v in batch])

    q_guess_prim = qn_guess_prim(np.expand_dims(outcome_1_beta1_batch, axis=0))
    q_guess_prim = np.squeeze(q_guess_prim.numpy())

    opt_a_2_prim = np.argmax(q_guess_prim,axis=1)

    update_for_q_1_prim = qn_l1_targ(np.expand_dims(np.array([[] for i in range(len(batch))]), axis=0)) #targ = target
    update_for_q_1_prim = np.squeeze(update_for_q_1_prim, axis=0)
    qlabels_l1 = update_for_q_1_prim.copy()

    # print(qlabels_l1.shape, type(qlabels_l1), batch_length, np.arange(batch_length))
    print(labels_beta1)
    np.save("labelsbeta1", labels_beta1)
    qlabels_l1[np.arange(batch_length), labels_beta1] = np.squeeze(qn_guess_targ(np.expand_dims(outcome_1_beta1_batch, axis=0)).numpy())[np.arange(batch_length),opt_a_2_prim]

    optimizer_ql1, optimizer_guess = optimizers

    train_vars_in = [ch.numpy() for ch in qn_l1_prim.trainable_variables]

    # print(qlabels_l1)
    with tf.device("/cpu:0"):
        with tf.GradientTape() as tape:
            tape.watch(qn_l1_prim.trainable_variables)
            pred_q_1s = qn_l1_prim(np.expand_dims(np.array([[] for i in range(len(batch))]), axis=0))

            loss_sum =tf.keras.losses.MSE(pred_q_1s, qlabels_l1)
            loss = tf.reduce_mean(loss_sum)
            grads = tape.gradient(loss, qn_l1_prim.trainable_variables)
            optimizer_ql1.apply_gradients(zip(grads, qn_l1_prim.trainable_variables))

        with tf.GradientTape() as tape:
            tape.watch(qn_guess_prim.trainable_variables)
            pred_q_3s = qn_guess_prim(np.expand_dims(s_3_batch, axis=0))
            loss_sum =tf.keras.losses.MSE(pred_q_3s, qlabels_l3)
            loss = tf.reduce_mean(loss_sum)
            loss_guess = loss
            grads = tape.gradient(loss, qn_guess_prim.trainable_variables)
            optimizer_guess.apply_gradients(zip(grads, qn_guess_prim.trainable_variables))

    for t, e in zip(qn_l1_targ.trainable_variables, qn_l1_prim.trainable_variables):
        t.assign(t*(1-TAU) + e*TAU)

    for t, e in zip(qn_guess_targ.trainable_variables, qn_guess_prim.trainable_variables):
        t.assign(t*(1-TAU) + e*TAU)
    return loss, loss_guess


losses= []
networks = [qn_l1_prim, qn_l1_targ, qn_guess_prim, qn_guess_targ]

qn_l1_prim.give_first_beta(epsilon=1)
qn_l1_targ.give_first_beta(epsilon=1)
qn_guess_prim.give_guess([0, -.7], epsilon=1)
qn_guess_targ.give_guess([0, -.7], epsilon=1)

learn_batch(networks, 1000)
