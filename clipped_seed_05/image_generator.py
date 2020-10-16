import numpy as np
from misc import Prob, Basics
import matplotlib.pyplot as plt
import argparse
from nets import RNNC
from glob import glob
from tqdm import tqdm

basics = Basics(dolinar_layers=2)
ats = basics.ats

def pn1(n,beta,alpha=0.4):
    p=0
    for pm in [-1,1]:
        p+=Prob(pm*alpha, beta,n)
    return p/2

def Q2(b1,n1,b2,alpha=0.4):
    p=0
    for n2 in [0,1]:
        p+=max([Prob(pm*alpha*np.cos(ats[0]), b1,n1)*Prob(pm*alpha*np.sin(ats[0]), b2,n2) for pm in [-1.,1.]])
    p/=pn1(n1,b1)
    return p/2

def Q1(b1,alpha=0.4):
    p=0
    for n1 in [0,1]:
        p+= max([Q2(b1,n1,b2,alpha) for b2 in np.arange(-1,1,.05)])*pn1(n1,b1)
    return p/2

betas = np.arange(-1,1,.05)
b1=-.5

stoppings = []
for k in range(4):
    for kk in range(10**k,10**(k+1),10**k):
        stoppings.append(kk)
for j in range(1,4):
    stoppings.append(j*10**4)

net = RNNC(return_sequences=False)
for epoch in tqdm(stoppings):
    net.load_weights("net/"+str(epoch))
    plt.figure(figsize=(10,10))
    plt.suptitle("epoch: {}".format(epoch), size=20)
    ax1=plt.subplot2grid((2,1),(0,0))
    ax2=plt.subplot2grid((2,1),(1,0))
    axs = {0:ax1,1:ax2}
    c=0
    data_test = {}
    for nn in [0.,1.]:
        data_test[str(nn)] = []
        for b in betas:
            data_test[str(nn)].append([[b1,-1.],[b,nn]])
        axs[c].plot(betas, np.squeeze(net(np.reshape(data_test[str(nn)], (len(data_test[str(nn)]), 2,2)))), alpha=0.7, color="blue", linewidth=5)
        axs[c].plot(betas, [Q2(b1,nn,b) for b in betas], '--', label="Q2", alpha=0.7, linewidth=5, color="red")
        c+=1

    plt.savefig("giyy_clip_seed/{}.png".format(epoch))
