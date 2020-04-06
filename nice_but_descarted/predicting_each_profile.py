def supervised_training(batch_size=32., total_epochs=30, buffer_size=10**4, tau=0.0005, lr=0.001):

    net_0 = Net(input_dim=1)
    net_1 = Net(input_dim=3)
    net_1_target = Net(input_dim=3)

    net_0(np.array([[0.],[1.]])) #initialize the network 0, arbitrary inputs.
    net_1(np.array([[0.,1.,1.]]))
    net_1_target(np.array([[0.,1.,1.]]))

    optimizer_0 = tf.keras.optimizers.Adam(lr=lr)
    optimizer_1 = tf.keras.optimizers.Adam(lr=lr)

    dataset = DataSet(size=10**4, nbetas=15)
    histo_preds = {"net_0":{}, "net_1":{}} #here i save the predictions


    train_loss_0 = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss_0 = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    train_loss_1 = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss_1 = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)



    for epoch_number in tqdm(range(total_epochs)):
        epoch = dataset.give_epoch(batch_size)


        for batch in epoch: #{a0}, {h1_a1}, {r2}
            labels_net0 = calculate_greedy_from_batch(net_1_target, batch ) #greedy from target!
            labels_net1 = np.expand_dims(batch[2],axis=1)
            with tf.GradientTape() as tape:
                tape.watch(net_0.trainable_variables)
                preds0 = net_0(np.expand_dims(batch[0],axis=1))
            loss_0 = tf.keras.losses.MSE(labels_net0,preds0)
            loss_0 = tf.reduce_mean(loss_0)
            train_loss_0(loss_0)
            grads0 = tape.gradient(loss_0, net_0.trainable_variables)
            optimizer_0.apply_gradients(zip(grads0, net_0.trainable_variables))

            test_label_y = np.expand_dims(np.array([ps_maxlik(b) for b in batch[0]]), axis=1)
            loss_y0 = tf.reduce_mean(tf.keras.losses.MSE(test_label_y, preds0))
            test_loss_0(loss_y0)

            with tf.GradientTape() as tape:
                tape.watch(net_1.trainable_variables)
                preds1 = net_1(batch[1])
            loss_1 = tf.reduce_mean(tf.keras.losses.MSE(labels_net1,preds1))
            train_loss_1(loss_1)
            grads1 = tape.gradient(loss_1, net_1.trainable_variables)
            optimizer_1.apply_gradients(zip(grads1, net_1.trainable_variables))

            ###### update target #####
            net_1_target.update_target_parameters(net_1, tau=tau)
            ###### update target #####

        ### save the average losses per epoch###
        mean_loss_0 = np.mean(loss_0_avg)
        mean_loss_1 = np.mean(loss_1_avg)
        loss_0_ev.append(mean_loss_0)
        loss_1_ev.append(mean_loss_1)

        ### save the network's prediction at each epoch ###
        for nett in ["net_0","net_1"]:

            histo_preds[nett][str(epoch_number)] ={}
            histo_preds[nett][str(epoch_number)]["epoch_number"] = epoch_number
            histo_preds[nett][str(epoch_number)]["values"] = {}

        histo_preds["net_0"][str(epoch_number)]["values"] = np.squeeze(net_0(np.expand_dims(dataset.betas,axis=1)))

        index=0
        for n1 in [0.,1.]:
            for guess in [-1.,1.]:
                foo =np.array([[b,n1,guess] for b in dataset.betas]) #betas_train defined as global in create_dataset_l2()
                histo_preds["net_1"][str(epoch_number)]["values"][str(index)] = np.squeeze(net_1(foo))
                index+=1
        if epoch_number%max(1,int(total_epochs/10)) ==0:
            #plot_predictions(histo_preds, [loss_0_ev, loss_1_ev], dataset.betas)
            print("#### \nepoch: {}\nloss_0: {}\nloss_1: {}\n \n***********\n".format(epoch_number,mean_loss_0,mean_loss_1))
    losses = [loss_0_ev, loss_1_ev]
    return histo_preds, losses, dataset.betas
