
# coding: utf-8

# In[157]:


import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
tf.keras.backend.set_floatx('float64')


def Prob(alpha, beta, n1):
    p0 = np.exp(-(alpha-beta)**2)
    if n1 == 0:
        return p0
    else:
        return 1-p0
def ps(beta):
    #dolinar guessing rule (= max-likelihood for L=1, careful sign of \beta)
    alpha = 0.4
    p=0
    for n1 in [0,1]:
       p+=Prob((-1)**(n1+1)*alpha, beta, n1)
    return p/2

class Q1(tf.keras.Model):
    def __init__(self):
        super(Q1,self).__init__()

        self.l1 = Dense(10, input_shape=(1,),kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))
        self.l2 = Dense(33, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))
        self.l3 = Dense(10, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))
        self.l4 = Dense(33, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))
        self.l5 = Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.relu(self.l4(feat))
        value = tf.nn.tanh(self.l5(feat))
        return value


def create_dataset(number_betas, reward_samples, fake_dataset=False):
    data = []
    betas = np.linspace(-1,1,number_betas)
    psucs = [ps(b) for b in betas]
    if fake_dataset==True:
        for i in range(len(betas)):
            for l in range(reward_samples):
                data.append(np.array([betas[i],psucs[i]]))
        return np.array(data), betas

    else:
        for i in range(len(betas)):
            p=0
            for l in range(reward_samples):
                r=np.random.choice([1,0], 1, p=[psucs[i], 1-psucs[i]])[0]
                p+=r
                data.append(np.array([betas[i],r]))
        return np.array(data), betas

    for i in range(len(betas)):
        p=0
        for l in range(reward_samples):
            r=np.random.choice([1,0], 1, p=[psucs[i], 1-psucs[i]])[0]
            p+=r
            data.append(np.array([betas[i],r]))
    return np.array(data), betas

    for i in range(len(betas)):
        p=0
        for l in range(reward_samples):
            r=np.random.choice([1,0], 1, p=[psucs[i], 1-psucs[i]])[0]
            p+=r
            data.append(np.array([betas[i],r]))
    return np.array(data), betas






#As we may do a lot of experiments, let us define a function to record the result into a folder!

#Don't worry, this is a bit trivial, only keeps a record of how many times we have run a given experiment, it's unimportant!

def record():
    if not os.path.exists("number_rune.txt"):
        with open("number_rune.txt", "w+") as f:
            f.write("0")
            f.close()
        a=0
        number_run=0
    else:
        with open("number_rune.txt", "r") as f:
            a = f.readlines()[0]
            f.close()
        with open("number_rune.txt", "w") as f:
            f.truncate(0)
            f.write(str(int(a)+1))
            f.close()
        number_run = int(a)+1
    if not os.path.exists("run_"+str(number_run)):
        os.makedirs("run_"+str(number_run))
    return number_run


# In[163]:


#Let us define yet another function to make the plots, given some "predictions" done by the net at after the final epoch, and the evolution of the loss.

def make_figure(run_id,lr, Nbatch, loss_train, loss_test, predictions, epochs, betas_train, shuffle="False"):
    name_title = "Learning rate: "+ str(lr)+ "\nNumber of batches: "+str(Nbatch)
    name="lr"+str(lr)+"_"+"nb"+str(Nbatch)
    if shuffle == "True":
        name = name+"_shuffled"
    plt.figure(figsize=(10,10))
    ax1=plt.subplot2grid((2,1),(0,0))
    ax2=plt.subplot2grid((2,1),(1,0))
    plt.subplots_adjust(hspace=0.5)
    plt.title(name_title)
    ax1.plot(np.log10(range(1,epochs+1)), loss_train, color="red", label="Training loss", alpha=0.7)
    ax1.plot(np.log10(range(1,epochs+1)), loss_test, color="blue", label="Testing loss", alpha=0.7)
    ax1.set_xlabel("epoch",size=25)
    ax1.legend()

    ax2.scatter(betas_test, predictions, s=20, color="red",label="Predictions",alpha=0.7)
    ax2.plot(betas_test, ps(betas_test), '--',color="blue",label="Testing range" ,alpha=0.7)
    ax2.scatter(betas_train, ps(betas_train), color="green", label="Training set",alpha=0.7 )

    ax2.set_xlabel(r'$\beta$',size=25)
    ax1.set_title("Loss evolution",size=25)
    ax2.set_title("Success probability: prediction vs. true", size=25)
    ax2.legend()
    plt.savefig(run_id+"/"+str(name)+".png")
    plt.close()
    return


# In[165]:


def training(run_id, lr=10**-3,Nbatch=10,epochs=10, shuffle=True,
             number_betas=10, reward_samples=1000, fake_dataset=False, n1=10, n2=33, method="Adam", net="Q1"):
    if net=="Q1":
        q1=Q1()
    else:
        q1=Qdeep()
    if method == "Adam":
        optimizer = tf.keras.optimizers.Adam(lr = lr)
    else:
        optimizer = tf.keras.optimizers.SGD(lr = lr)

    datashu, betas_train = create_dataset(number_betas, reward_samples, fake_dataset)
    np.random.shuffle(datashu)
    databatches = np.split(datashu, Nbatch)

    loss_train = []
    loss_test=[]
    preds = []
    for epoch in tqdm(range(epochs)):
        # if shuffle==True:
        #     np.random.shuffle(datashu)
        #     databatches = np.split(datashu, Nbatch)
        for i in range(Nbatch):
            betas = databatches[i][:,0]
            rewards = databatches[i][:,1]
            with tf.device("/cpu:0"):
                with tf.GradientTape() as tape:
                    tape.watch(q1.trainable_variables)
                    predictions = q1(np.expand_dims(betas,axis=1))
                    loss_sum = tf.keras.losses.MSE(predictions,np.expand_dims(rewards,axis=1))
                    loss = tf.reduce_mean(loss_sum)
                    grads = tape.gradient(loss, q1.trainable_variables)
                    optimizer.apply_gradients(zip(grads, q1.trainable_variables))

        predictions = q1(np.expand_dims(betas_test,axis=1))
        loss_sum_outside = tf.keras.losses.MSE(predictions,np.expand_dims(ps(betas_test),axis=1))
        loss_test_ev = tf.reduce_mean(loss_sum_outside)

        loss_test.append(loss_test_ev.numpy())
        loss_train.append(loss.numpy())

        pp = np.squeeze(q1(np.expand_dims(betas_test,axis=1)).numpy())
        preds.append(pp)

        name="lr"+str(lr)+"_"+"nb"+str(Nbatch)
    np.save(run_id+"/loss_"+name,loss_train)
    np.save(run_id+"/loss_"+name,loss_test)

    np.save(run_id+"/preds_"+name,preds, allow_pickle=True)

    test_predictions = np.squeeze(q1(np.expand_dims(betas_test,axis=1)).numpy())
    make_figure(run_id,lr, Nbatch, loss_train, loss_test, predictions, epochs, betas_train,shuffle=shuffle)

    data = "Data set size: {}\nlearning rate: {}\nNumber of betas: {}\nNumber of samples per beta: {}\nBatch size: {}\nMethod: {}\nEpochs: {}\nShuffling: {}".format(
        number_betas*reward_samples, lr, number_betas, reward_samples, Nbatch, optimizer.__str__(), epochs, shuffle)
    data = data + "Fake dataset: {}\nneurons layer 1: {}\nneurons layer 2:{}".format(fake_dataset, n1,n2)
    os.chdir(str(run_id))
    with open("info.txt", 'w') as f:
        f.write(data)
        f.close()
    os.chdir("..")
    return


if not os.path.exists("stoch"):
    os.makedirs("stoch")
os.chdir("stoch")

betas_test = np.arange(-1,1.1,.01)
number_betas = 20
reward_samples = 1000


number_run = record()
training("run_"+str(number_run),lr=10**-2,Nbatch=5, epochs=10**2,
         shuffle=True, number_betas=number_betas, reward_samples=reward_samples, fake_dataset=False, method="Adam")
# number_run = record()
# training("run_"+str(number_run),lr=0.5*10**-3,Nbatch=batchi, epochs=10**3,
#          shuffle=False, number_betas=number_betas, reward_samples=reward_samples, fake_dataset=True)
