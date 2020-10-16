import numpy as np
from misc import Prob, Basics
import matplotlib.pyplot as plt
import argparse
from nets import RNNC
from glob import glob

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
net = RNNC(return_sequences=False)

epoch = len(glob("giffy"))
net.load_weights("nets/"+str(args.run)+"/")
b1=-.5
plt.figure(figsize=(10,10))
plt.suptitle("epoch: {}".format(epoch), size=20)
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
    axs[c].plot(betas, np.squeeze(net(np.reshape(data_test[str(nn)], (len(data_test[str(nn)]), 2,2)))), alpha=0.7, color="blue", linewidth=5)
    axs[c].plot(betas, [Q2(b1,nn,b) for b in betas], '--', label="Q2", alpha=0.7, linewidth=5, color="red")
    c+=1

plt.savefig("giffy/{}.png".format(epoch))
