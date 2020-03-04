import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import deque
import random



# def make_figure(run_id,lr, Nbatch, loss_train, loss_test, predictions,
#  epochs, betas_train, shuffle="False"):
#     name_title = "Learning rate: "+ str(lr)+ "\nNumber of batches: "+str(Nbatch)
#     name="lr"+str(lr)+"_"+"nb"+str(Nbatch)
#     if shuffle == "True":
#         name = name+"_shuffled"
#     plt.figure(figsize=(10,10))
#     ax1=plt.subplot2grid((2,1),(0,0))
#     ax2=plt.subplot2grid((2,1),(1,0))
#     plt.subplots_adjust(hspace=0.5)
#     plt.title(name_title)
#     ax1.plot(np.log10(range(1,epochs+1)), loss_train, color="red", label="Training loss", alpha=0.7)
#     ax1.plot(np.log10(range(1,epochs+1)), loss_test, color="blue", label="Testing loss", alpha=0.7)
#     ax1.set_xlabel("epoch",size=25)
#     ax1.legend()
#
#     ax2.scatter(betas_test, predictions, s=20, color="red",label="Predictions",alpha=0.7)
#     ax2.plot(betas_test, ps(betas_test), '--',color="blue",label="Testing range" ,alpha=0.7)
#     ax2.scatter(betas_train, ps(betas_train), color="green", label="Training set",alpha=0.7 )
#
#     ax2.set_xlabel(r'$\beta$',size=25)
#     ax1.set_title("Loss evolution",size=25)
#     ax2.set_title("Success probability: prediction vs. true", size=25)
#     ax2.legend()
#     plt.savefig(run_id+"/"+str(name)+".png")
#     plt.close()
#     return

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


def check_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)
    os.chdir(name)



def greedy_action(q1, betas, ep=0):
    if np.random.random()<ep:
        l = np.random.choice(range(len(betas)), 1)[0]
        return l, betas[l]
    else:
        qs= np.squeeze(q1(np.expand_dims(betas, axis=1)).numpy())
        l=np.where(qs==max(qs))[0]
        # print(qs)
        # print(ps(betas))
        # print("***")
        if len(l)>1:
            l=np.random.choice(l,1)[0]
        else:
            l=l[0]
        return l, betas[l]




class ReplayBuffer:
    def __init__(self, buffer_size=10**3):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, a, r):
        experience = (a,r)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        a_batch, r_batch= list(map(np.array, list(zip(*batch))))
        return a_batch, r_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


def plot_evolution(rt, pt, optimal, betas, preds, loss=False, history_betas=False,run_id="algo"):
    matplotlib.rc('font', serif='cm10')
    plt.rcParams.update({'font.size': 50})

    if loss != "False":
        plt.figure(figsize=(40,40), dpi=100)
        T=len(rt)
        ax1=plt.subplot2grid((2,2),(0,0))
        ax2=plt.subplot2grid((2,2),(1,0))
        ax3=plt.subplot2grid((2,2),(0,1))
        ax4=plt.subplot2grid((2,2),(1,1))

        ax1.plot(np.log10(np.arange(1,T+1)),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
        ax1.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
        ax2.plot(np.log10(np.arange(1,T+1)),pt, color="red", linewidth=15, alpha=0.8, label=r'$P_t$')
        ax2.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
        ax3.scatter(betas, preds, color="red", s=250, label="predictions", alpha=0.6)
        ax3.scatter(betas, ps(betas), color="blue", s=250, label="true values", alpha=0.6)
        ax4.plot(loss, color="black",  linewidth=15,alpha=0.5, label="Loss evolution")

        for ax in [ax1, ax2, ax3,ax4]:
            ax.legend(prop={"size":30})
        plt.savefig(run_id+"/learning_curves.png")
        plt.close()

        if history_betas!=False:
            optimal_beta = betas[np.where(ps(betas) == max(ps(betas)))[0][0]]
            plt.figure(figsize=(40,40), dpi=100)

            plt.hist(history_betas,density=True, facecolor='r', alpha=0.75, edgecolor='blue')
            plt.text(optimal_beta, 0, "*", size=50)
            plt.savefig(run_id+"/histograma_betas.png")
            plt.close()
        return
    else:
        plt.figure(figsize=(30,15))
        T=len(rt)
        ax1=plt.subplot2grid((2,2),(0,0))
        ax2=plt.subplot2grid((2,2),(1,0))
        ax3=plt.subplot2grid((2,2),(0,1), rowspan=2)
        ax1.plot(np.arange(1,T+1),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
        ax1.plot(optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
        ax2.plot(np.arange(1,T+1),pt, color="red", linewidth=15, alpha=0.8, label=r'$P_t$')
        ax2.plot(optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
        ax3.scatter(betas, preds, color="red", s=250, label="predictions", alpha=0.6)
        ax3.scatter(betas, ps(betas), color="blue", s=250, label="true values", alpha=0.6)

        for ax in [ax1, ax2, ax3]:
            ax.legend(prop={"size":30})
        plt.savefig(run_id+"/learning_curves.png")
        plt.close()
        return
