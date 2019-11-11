import tensorflow as tf
import numpy as np
import os

def learn(networks, optimizers, buffer, batch_length=1000, TAU=0.01, episode_info=0):
    if buffer.num_samples<1000:
        return
    else:

        batch = buffer.sample(batch_length)

        qn_l1_prim, qn_l1_targ, qn_guess_prim, qn_guess_targ = networks

        outcome_1_beta1_batch = np.array([[ v[0], v[1]] for v in batch ] )
        print(outcome_1_beta1_batch)
        labels_beta1 = np.array([v[2] for v in batch])

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

        # QUESTIONS: 1) SHOULD I RETURN THE NETWORRKS AND LOAD THEM AGAIN?
        # 2)WHY THIS IS NOT BEING UPDATED?

        if episode_info%400==1:

            for i,j in zip(train_vars_in, qn_l1_prim.trainable_variables):

                with open("diff_learning/"+ str(episode_info) + "-" + str(j.name).replace("/",""), "w") as f:
                    f.write(str((i-j.numpy())))
                    f.close()

            with open("diff_learning/grads_"+str(episode_info) , "w") as f:
                f.write(str(grads))
                f.close()

            with open("diff_learning/loss_"+str(episode_info) , "w") as f:
                f.write(str(loss.numpy()))
                f.close()
                # np.save("diff_learning/"+ str(episode_info) + "-" + str(i.name).replace("/",""), (i-j).numpy())


        s_3_batch = np.array([[v[0], v[1] ] for v in batch])
        rewards = np.array([v[-1] for v in batch])
        labels_guess = np.array([v[-2] for v in batch])

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

        for t, e in zip(qn_guess_targ.trainable_variables, qn_guess_prim.trainable_variables):
            t.assign(t*(1-TAU) + e*TAU)
        return
