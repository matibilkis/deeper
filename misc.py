import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm as tqdm
tf.keras.backend.set_floatx('float64')
from collections import deque
from datetime import datetime
import random
import matplotlib




#### this is the outcome probability, given by the overlap <0|\alpha - \beta>|¨^{2}
def Prob(alpha, beta, n):
    p0 = np.exp(-(alpha-beta)**2)
    if n == 0:
        return p0
    else:
        return 1-p0

### this is just p(R=1 | g, n; beta) = p((-1^{g} alpha | n)) = p(n|allpha) pr(alpha)(p(n))
def qval(beta, n, guess):
    #dolinar guessing rule (= max-likelihood for L=1, careful sign of \beta)
    alpha = 0.4
    pn = np.sum([Prob(g*alpha, beta, n) for g in [-1,1]])
    return Prob(guess*alpha, beta, n)/pn

def ps_maxlik(beta):
    #dolinar guessing rule (= max-likelihood for L=1, careful sign of \beta)
    alpha = 0.4
    p=0
    for n1 in [0,1]:
       p+=Prob(np.sign(beta)*(-1)**(n1)*alpha, beta, n1)
    return p/2


def record():
    if not os.path.exists("results/number_rune.txt"):
        with open("results/number_rune.txt", "w+") as f:
            f.write("0")
            f.close()
        a=0
        number_run=0
    else:
        with open("results/number_rune.txt", "r") as f:
            a = f.readlines()[0]
            f.close()
        with open("results/number_rune.txt", "w") as f:
            f.truncate(0)
            f.write(str(int(a)+1))
            f.close()
        number_run = int(a)+1
    # if not os.path.exists("run_"+str(number_run)):
    #     os.makedirs("results/run_"+str(number_run))
    #
    # if not os.path.exists("results/run_"+str(number_run)+"/models"):
    #     os.makedirs("results/run_"+str(number_run)+"/models")
    return number_run







#
# # def make_figure(run_id,lr, Nbatch, loss_train, loss_test, predictions,
# #  epochs, betas_train, shuffle="False"):
# #     name_title = "Learning rate: "+ str(lr)+ "\nNumber of batches: "+str(Nbatch)
# #     name="lr"+str(lr)+"_"+"nb"+str(Nbatch)
# #     if shuffle == "True":
# #         name = name+"_shuffled"
# #     plt.figure(figsize=(10,10))
# #     ax1=plt.subplot2grid((2,1),(0,0))
# #     ax2=plt.subplot2grid((2,1),(1,0))
# #     plt.subplots_adjust(hspace=0.5)
# #     plt.title(name_title)
# #     ax1.plot(np.log10(range(1,epochs+1)), loss_train, color="red", label="Training loss", alpha=0.7)
# #     ax1.plot(np.log10(range(1,epochs+1)), loss_test, color="blue", label="Testing loss", alpha=0.7)
# #     ax1.set_xlabel("epoch",size=25)
# #     ax1.legend()
# #
# #     ax2.scatter(betas_test, predictions, s=20, color="red",label="Predictions",alpha=0.7)
# #     ax2.plot(betas_test, ps(betas_test), '--',color="blue",label="Testing range" ,alpha=0.7)
# #     ax2.scatter(betas_train, ps(betas_train), color="green", label="Training set",alpha=0.7 )
# #
# #     ax2.set_xlabel(r'$\beta$',size=25)
# #     ax1.set_title("Loss evolution",size=25)
# #     ax2.set_title("Success probability: prediction vs. true", size=25)
# #     ax2.legend()
# #     plt.savefig(run_id+"/"+str(name)+".png")
# #     plt.close()
# #     return
#

#
#
# def create_dataset(number_betas, reward_samples, fake_dataset=False):
#     data = []
#     betas = np.linspace(-1,1,number_betas)
#     psucs = [ps(b) for b in betas]
#     if fake_dataset==True:
#         for i in range(len(betas)):
#             for l in range(reward_samples):
#                 data.append(np.array([betas[i],psucs[i]]))
#         return np.array(data), betas
#
#     else:
#         for i in range(len(betas)):
#             p=0
#             for l in range(reward_samples):
#                 r=np.random.choice([1,0], 1, p=[psucs[i], 1-psucs[i]])[0]
#                 p+=r
#                 data.append(np.array([betas[i],r]))
#         return np.array(data), betas
#
#     for i in range(len(betas)):
#         p=0
#         for l in range(reward_samples):
#             r=np.random.choice([1,0], 1, p=[psucs[i], 1-psucs[i]])[0]
#             p+=r
#             data.append(np.array([betas[i],r]))
#     return np.array(data), betas
#
#     for i in range(len(betas)):
#         p=0
#         for l in range(reward_samples):
#             r=np.random.choice([1,0], 1, p=[psucs[i], 1-psucs[i]])[0]
#             p+=r
#             data.append(np.array([betas[i],r]))
#     return np.array(data), betas
#
#
#
#
# def Prob(alpha, beta, n1):
#     p0 = np.exp(-(alpha-beta)**2)
#     if n1 == 0:
#         return p0
#     else:
#         return 1-p0
#
#
#
# def give_outcome(alpha,beta):
#     p0=np.exp(-(alpha-beta)**2)
#     out = np.random.choice([0,1],1,p=[p0,1-p0])[0]
#     return out
#
# def ps_maxlik(beta):
#     #dolinar guessing rule (= max-likelihood for L=1, careful sign of \beta)
#     alpha = 0.4
#     p=0
#     for n1 in [0,1]:
#        p+=Prob(np.sign(beta)*(-1)**(n1)*alpha, beta, n1)
#     return p/2
#
#
# def ps(beta,g):
#     #g = [guess|0, guess|1]
#     #dolinar guessing rule (= max-likelihood for L=1, careful sign of \beta)
#     alpha = 0.4
#     p=0
#     for n1 in [0,1]:
#        p+=Prob(g[n1]*alpha, beta, n1)
#     return p/2
#
#
#
#
#
# def check_folder(name):
#     if not os.path.exists(name):
#         os.makedirs(name)
#     os.chdir(name)
#
#
#
# def greedy_action(q1, betas, ep=0):
#     if np.random.random()<ep:
#         l = np.random.choice(range(len(betas)), 1)[0]
#         return l, betas[l]
#     else:
#         qs= np.squeeze(q1(np.expand_dims(betas, axis=1)).numpy())
#         l=np.where(qs==max(qs))[0]
#         # print(qs)
#         # print(ps(betas))
#         # print("***")
#         if len(l)>1:
#             l=np.random.choice(l,1)[0]
#         else:
#             l=l[0]
#         return l, betas[l]
#
#
# def gredy_action(q2, inp):
#     qval = np.squeeze(q2(np.expand_dims(inp, axis=0)).numpy())
#     l=np.where(qs==max(qs))[0]
#     if len(l)>1:
#         l=np.random.choice(l,1)[0]
#     else:
#         l=l[0]
#     return l, [-1,1][l]
#
#
# class ReplayBufferv1:
#     def __init__(self, buffer_size=10**3):
#         self.buffer_size = buffer_size
#         self.count = 0
#         self.buffer = deque()
#
#     def add(self, a, r):
#         experience = (a,r)
#         if self.count < self.buffer_size:
#             self.buffer.append(experience)
#             self.count += 1
#         else:
#             self.buffer.popleft()
#             self.buffer.append(experience)
#
#     def size(self):
#         return self.count
#
#     def sample(self, batch_size):
#         batch = []
#         if self.count < batch_size:
#             batch = random.sample(self.buffer, self.count)
#         else:
#             batch = random.sample(self.buffer, batch_size)
#         a_batch, r_batch= list(map(np.array, list(zip(*batch))))
#         return a_batch, r_batch
#
#     def clear(self):
#         self.buffer.clear()
#         self.count = 0
#
#
#
# class ReplayBufferKennedy:
#     def __init__(self, buffer_size=10**3):
#         self.buffer_size = buffer_size
#         self.count = 0
#         self.buffer = deque()
#
#     def add(self, a, n, g, r):
#         experience = (a, n , g, r)
#         if self.count < self.buffer_size:
#             self.buffer.append(experience)
#             self.count += 1
#         else:
#             self.buffer.popleft()
#             self.buffer.append(experience)
#
#     def size(self):
#         return self.count
#
#     def sample(self, batch_size):
#         batch = []
#         if self.count < batch_size:
#             batch = random.sample(self.buffer, self.count)
#         else:
#             batch = random.sample(self.buffer, batch_size)
#         a_batch, outcome_batch, guess_batch, r_batch= list(map(np.array, list(zip(*batch))))
#         return a_batch, outcome_batch, guess_batch, r_batch
#
#     def clear(self):
#         self.buffer.clear()
#         self.count = 0
#
#
# def save_models(run_id, models):
#     for k in models:
#         k.save_weights(run_id+"/models/"+k.__str__())
#
# def plot_evolution_vKennedy1(rt, pt, optimal, betas, preds=None, loss=False,
# history_betas=False, history_would_have_done=False,run_id="algo"):
#
#     matplotlib.rc('font', serif='cm10')
#     plt.rcParams.update({'font.size': 50})
#
#     plt.figure(figsize=(40,40), dpi=100)
#     T=len(rt)
#     ax1=plt.subplot2grid((2,2),(0,0))
#     ax2=plt.subplot2grid((2,2),(1,0))
#     ax3=plt.subplot2grid((2,2),(0,1))
#     ax4=plt.subplot2grid((2,2),(1,1))
#
#     ax1.plot(np.log10(np.arange(1,T+1)),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
#     ax1.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#     ax2.plot(np.log10(np.arange(1,T+1)),pt, color="red", linewidth=15, alpha=0.8, label=r'$P_t$')
#     ax2.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#
#     if preds!=None:
#         ax3.scatter(betas, preds, color="red", s=250, label="predictions", alpha=0.6)
#
#     ax3.scatter(betas, ps_maxlik(betas), color="blue", s=250, label="true values", alpha=0.6)
#     ax4.plot(loss, color="black",  linewidth=15,alpha=0.5, label="Loss evolution")
#
#     for ax in [ax1, ax2, ax3,ax4]:
#         ax.legend(prop={"size":30})
#     plt.savefig(run_id+"/learning_curves.png")
#     plt.close()
#
#     if history_betas!=False:
#         optimal_beta = betas[np.where(ps_maxlik(betas) == max(ps_maxlik(betas)))[0][0]]
#         plt.figure(figsize=(40,40), dpi=100)
#
#         plt.hist(history_betas,bins=100, facecolor='r', alpha=0.6, edgecolor='blue', label="done")
#         plt.hist(history_would_have_done,bins=100, facecolor='g', alpha=0.4, edgecolor='black', label="would have done")
#         plt.legend()
#         plt.text(optimal_beta, 0, "*", size=350)
#         plt.text(-optimal_beta, 0, "*", size=350)
#
#         plt.savefig(run_id+"/histograma_betas.png")
#         plt.close()
#
#
#
#         plt.figure(figsize=(40,40), dpi=100)
#         T=len(rt)
#         ax1=plt.subplot2grid((1,1),(0,0))
#
#         ax1.plot(np.arange(1, len(history_betas)+1),history_betas, color="red", linewidth=15, alpha=0.8, label="done")
#         ax1.plot(np.arange(1, len(history_betas)+1),history_would_have_done, color="green", linewidth=15, alpha=0.8, label="would have done")
#         ax1.plot(np.arange(1, len(history_betas)+1),np.ones(len(history_betas))*optimal_beta, color="black", linewidth=15, alpha=0.8, label="optimal-beta")
#         ax1.plot(np.arange(1, len(history_betas)+1),-np.ones(len(history_betas))*optimal_beta, color="black", linewidth=15, alpha=0.8, label="optimal-beta")
#
#
#         for ax in [ax1]:
#             ax.legend(prop={"size":30})
#         plt.savefig(run_id+"/evolution_actions.png")
#         plt.close()
#     return
#
# def qguess(b,n1,sign=1):
#     alpha=0.4
#     pn1 = np.sum([Prob(ph*alpha, b, n1) for ph in [-1,1]])
#     return Prob(sign*alpha, b, n1)/pn1
#
#
# def plot_evolution_vKennedy_last(rt, pt, optimal, betas, preds=None, pred_guess=None, loss=False,
# history_betas=False, history_would_have_done=False,run_id="algo"):
#
#     matplotlib.rc('font', serif='cm10')
#     plt.rcParams.update({'font.size': 50})
#
#     plt.figure(figsize=(40,40), dpi=100)
#     T=len(rt)
#     ax1=plt.subplot2grid((2,3),(0,0))
#     ax2=plt.subplot2grid((2,3),(1,0))
#     ax3=plt.subplot2grid((2,3),(0,1))
#     ax4=plt.subplot2grid((2,3),(1,1))
#     ax5=plt.subplot2grid((2,3),(0,2))
#     ax6=plt.subplot2grid((2,3),(1,2))
#
#
#     ax1.plot(np.log10(np.arange(1,T+1)),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
#     ax1.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#     ax2.plot(np.log10(np.arange(1,T+1)),pt, color="red", linewidth=15, alpha=0.8, label=r'$P_t$')
#     ax2.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#
#     ax3.scatter(betas, preds, color="red", s=250, label="predictions", alpha=0.6)
#     ax3.scatter(betas, ps_maxlik(betas), color="blue", s=250, label="true values", alpha=0.6)
#     ax4.plot(loss, color="black",  linewidth=15,alpha=0.5, label="Loss evolution")
#
#     pred_guess_0, pred_guess_1 = pred_guess
#     ax5.plot(betas, pred_guess_0[:,0], color="red", linewidth=15, alpha=0.8, label=r'$\hat{Q}((\beta,0), +)$')
#     ax5.plot(betas, [qguess(b,0,sign=1) for b in betas], color="blue", linewidth=15, alpha=0.8, label=r'$Q((\beta,0), +)$')
#     ax5.plot(betas, pred_guess_1[:,0], color="red", linewidth=15, alpha=0.8, label=r'$\hat{Q}((\beta, 1), +)$')
#     ax5.plot(betas, [qguess(b,1,sign=1) for b in betas], color="blue", linewidth=15, alpha=0.8, label=r'$Q((\beta, 1) +)$')
#
#     ax6.plot(betas, pred_guess_0[:,1], color="red", linewidth=15, alpha=0.8, label=r'$\hat{Q}((\beta,0), -)$')
#     ax6.plot(betas, [qguess(b,0,sign=-1) for b in betas], color="blue", linewidth=15, alpha=0.8, label=r'$Q((\beta,0), -)$')
#     ax6.plot(betas, pred_guess_1[:,1], color="red", linewidth=15, alpha=0.8, label=r'$\hat{Q}((\beta, 1), -)$')
#     ax6.plot(betas, [qguess(b,1,sign=-1) for b in betas], color="blue", linewidth=15, alpha=0.8, label=r'$Q((\beta, 1) -)$')
#
#     for ax in [ax1, ax2, ax3,ax4,ax5,ax6]:
#         ax.legend(prop={"size":30})
#     plt.savefig(run_id+"/learning_curves.png")
#     plt.close()
#
#     if history_betas!=False:
#         optimal_beta = betas[np.where(ps_maxlik(betas) == max(ps_maxlik(betas)))[0][0]]
#         plt.figure(figsize=(50,50), dpi=100)
#
#         plt.hist(history_betas,bins=100, facecolor='r', alpha=0.6, edgecolor='blue', label="done")
#         plt.hist(history_would_have_done,bins=100, facecolor='g', alpha=0.4, edgecolor='black', label="would have done")
#         plt.legend()
#         plt.text(optimal_beta, 0, "*", size=350)
#         plt.text(-optimal_beta, 0, "*", size=350)
#
#         plt.savefig(run_id+"/histograma_betas.png")
#         plt.close()
#
#
#
#         plt.figure(figsize=(40,40), dpi=100)
#         T=len(rt)
#         ax1=plt.subplot2grid((1,1),(0,0))
#
#         ax1.plot(np.arange(1, len(history_betas)+1),history_betas, color="red", linewidth=15, alpha=0.8, label="done")
#         ax1.plot(np.arange(1, len(history_betas)+1),history_would_have_done, color="green", linewidth=15, alpha=0.8, label="would have done")
#         ax1.plot(np.arange(1, len(history_betas)+1),np.ones(len(history_betas))*optimal_beta, color="black", linewidth=15, alpha=0.8, label="optimal-beta")
#         ax1.plot(np.arange(1, len(history_betas)+1),-np.ones(len(history_betas))*optimal_beta, color="black", linewidth=15, alpha=0.8, label="optimal-beta")
#
#
#         for ax in [ax1]:
#             ax.legend(prop={"size":30})
#         plt.savefig(run_id+"/evolution_actions.png")
#         plt.close()
#
#
#     return
#
#
#
# def plot_evolution_v1(rt, pt, optimal, betas, preds=None, loss=False,
# history_betas=False, history_would_have_done=False,run_id="algo"):
#     matplotlib.rc('font', serif='cm10')
#     plt.rcParams.update({'font.size': 50})
#
#     if loss != "False":
#         plt.figure(figsize=(40,40), dpi=100)
#         T=len(rt)
#         ax1=plt.subplot2grid((2,2),(0,0))
#         ax2=plt.subplot2grid((2,2),(1,0))
#         ax3=plt.subplot2grid((2,2),(0,1))
#         ax4=plt.subplot2grid((2,2),(1,1))
#
#         ax1.plot(np.log10(np.arange(1,T+1)),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
#         ax1.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#         ax2.plot(np.log10(np.arange(1,T+1)),pt, color="red", linewidth=15, alpha=0.8, label=r'$P_t$')
#         ax2.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#         if preds!=None:
#             ax3.scatter(betas, preds, color="red", s=250, label="predictions", alpha=0.6)
#         ax3.scatter(betas, ps(betas), color="blue", s=250, label="true values", alpha=0.6)
#         ax4.plot(loss, color="black",  linewidth=15,alpha=0.5, label="Loss evolution")
#
#         for ax in [ax1, ax2, ax3,ax4]:
#             ax.legend(prop={"size":30})
#         plt.savefig(run_id+"/learning_curves.png")
#         plt.close()
#
#         if history_betas!=False:
#             optimal_beta = betas[np.where(ps(betas) == max(ps(betas)))[0][0]]
#             plt.figure(figsize=(40,40), dpi=100)
#
#             plt.hist(history_betas,bins=100, facecolor='r', alpha=0.6, edgecolor='blue', label="done")
#             plt.hist(history_would_have_done,bins=100, facecolor='g', alpha=0.4, edgecolor='black', label="would have done")
#             plt.legend()
#             plt.text(optimal_beta, 0, "*", size=350)
#             plt.text(-optimal_beta, 0, "*", size=350)
#
#             plt.savefig(run_id+"/histograma_betas.png")
#             plt.close()
#         return
#     else:
#         plt.figure(figsize=(30,15))
#         T=len(rt)
#         ax1=plt.subplot2grid((2,2),(0,0))
#         ax2=plt.subplot2grid((2,2),(1,0))
#         ax3=plt.subplot2grid((2,2),(0,1), rowspan=2)
#         ax1.plot(np.arange(1,T+1),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
#         ax1.plot(optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#         ax2.plot(np.arange(1,T+1),pt, color="red", linewidth=15, alpha=0.8, label=r'$P_t$')
#         ax2.plot(optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#         if preds!=None:
#             ax3.scatter(betas, preds, color="red", s=250, label="predictions", alpha=0.6)
#         ax3.scatter(betas, ps(betas), color="blue", s=250, label="true values", alpha=0.6)
#
#         for ax in [ax1, ax2, ax3]:
#             ax.legend(prop={"size":30})
#         plt.savefig(run_id+"/learning_curves.png")
#         plt.close()
#         return
