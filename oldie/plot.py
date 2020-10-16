import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
#from IPython import display
#%load_ext autoreload
#%autoreload 2


from misc import Prob, Basics
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--run", type=int, default=0)
args = parser.parse_args()

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

from nets import RNNC
net = RNNC(return_sequences=False)


vals = {}
# c=0
# for lr in [0.01, 0.001]:
#     for bs in [8, 64, 128, 512]:
#         vals[str(c)] = str([lr, bs])
#         c+=1
# vals["9"]=[2,2]

# for k in range(len(list(vals.keys())[0:2])):
for k in [args.run]:
    net.load_weights("nets/"+str(k)+"/")
    b1=-.5
    plt.figure(figsize=(10,10))
    ax1=plt.subplot2grid((2,1),(0,0))
    ax2=plt.subplot2grid((2,1),(1,0))
    #ax3=plt.subplot2grid((2,2),(0,1),rowspan=2)

    # ]plt.suptitle(vals[str(k)],size=20)
    axs = {0:ax1,1:ax2}
    c=0
    data_test = {}
    for nn in [0.,1.]:
        data_test[str(nn)] = []
        for b in betas:
            data_test[str(nn)].append([[b1,-1.],[b,nn]])
        axs[c].plot(betas, np.squeeze(net(np.reshape(data_test[str(nn)], (len(data_test[str(nn)]), 2,2)))), alpha=0.7)
        axs[c].plot(betas, [Q2(b1,nn,b) for b in betas], '--', label="Q2")
        c+=1
    plt.show()
    plt.close()
        #display.display(plt.gcf())
    #ax3.plot(l, color="black", linewidth=9, alpha=.8)
    #ax3.plot(range(len(lt)), lt,color="blue", linewidth=9, alpha=.8, label="globl")
