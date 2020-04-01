class DataSet():
    def __init__(self, rpbgo=10, nbetas=10):
        #rpbgo = rewards per beta-guess-outcome
        self.nbetas=nbetas
        self.rpbgo=rpbgo
        self.betas = np.linspace(-1.5,1.5,nbetas)
        self.size = self.nbetas*self.rpbgo*4.
        d = []
        for b in self.betas:
            for outcome in [0.,1,]:
                for guess in [-1.,1.]:
                    for k in range(self.rpbgo):
                        mean_bernoulli = qval(b, outcome, guess)
                        reward = np.random.choice([1.,0,],1,p=[mean_bernoulli, 1.-mean_bernoulli])[0]

                        d.append([b, outcome, guess, reward])
        self.data_unsplitted = np.array(d)

    def batched_shuffled_dataset(self,splits):
        datacopy = self.data_unsplitted.copy()
        np.random.shuffle(datacopy)
        datacopy = np.split(datacopy, splits + len(datacopy)%splits)
        return datacopy


    def split_dataset(self, batch_size):
        dataset = self.data_unsplitted.copy()
        splits = int(len(dataset)/batch_size)
        if len(dataset)%batch_size !=0:
            #print("Not divisible!: breaking into len(dataset)%batch_size")
            splits = int(len(dataset)/batch_size)
            sobra = int(len(dataset)%batch_size)
            splited = np.split(dataset[sobra:], splits + len(dataset[sobra:])%splits)
            return splited, len(splited[0])
        else:
            splited = np.split(dataset, splits + len(dataset)%splits)
            return splited, len(splited[0])




def training(splits_over_size=1.):
    net = Net()
    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    dataset = DataSet(rpbgo=10,nbetas=10)
    batched_dataset = dataset.batched_shuffled_dataset(dataset.size/splits_over_size)

    net(batched_dataset[0][:,[0,1,2]]) #initialize the network

    print("batch_size: ",len(batched_dataset[0]))
    histo_preds = {} #here i save the predictions
    loss_ev = []

    ee={}
    ee["0"]=[]
    for k in net.trainable_variables:
        ee["0"].append(k.numpy())

    epochs=10
    for epoch in tqdm(range(epochs*splits_over_size)):
        losses=[]
        for mini_batch in batched_dataset:
            with tf.GradientTape() as tape:
                tape.watch(net.trainable_variables)
                preds = net(mini_batch[:,[0,1,2]])
                rew = mini_batch[:,3]
                loss = tf.keras.losses.MSE(rew,preds)
                loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, net.trainable_variables)
                optimizer.apply_gradients(zip(grads, net.trainable_variables))
                losses.append(loss)
        loss_ev.append(np.mean(np.array(losses)))

        batched_dataset = dataset.batched_shuffled_dataset(dataset.size/splits_over_size) #reshuffle

        ee[str(epoch+1)]=[]
        for k in net.trainable_variables:
            ee[str(epoch+1)].append(k.numpy())

        if (epoch % int(max(1,epochs/3)) == 0)|(epoch==epochs-1):
                histo_preds[str(epoch)] ={}
                histo_preds[str(epoch)]["epoch_number"] = epoch
                histo_preds[str(epoch)]["values"] = {}

                index=0
                for n1 in [0.,1.]:
                    for guess in [-1.,1.]:
                        foo =np.array([[b,n1,guess] for b in dataset.betas]) #betas_train defined as global in create_dataset_l2()
                        histo_preds[str(epoch)]["values"][str(index)] = np.squeeze(net(foo))
                        index+=1

    differences=[]
    for e1,e2 in zip(ee["0"],ee[str(epochs)]):
        differences.append(np.mean((e1-e2)/np.array(e1)))

    dataavg = np.split(dataset, len(betas_train))
    mean_values = {}
    for index_beta in range(len(betas_train)):
        mean_values[str(index_beta)] = {}
    for index_beta, beta in enumerate(betas_train):
        sp = np.split(dataavg[index_beta],4)
        for index_ng in range(4):
            mean_values[str(index_beta)][str(index_ng)]=np.mean(sp[index_ng][:,3])
    return loss_ev, histo_preds, mean_values, differences
