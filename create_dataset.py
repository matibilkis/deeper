import numpy as np
import os


from misc import Prob, Basics
import matplotlib.pyplot as plt
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

def Q1_star(b1,alpha=0.4):
    p=0
    for n1 in [0,1]:
        p+= max([Q2(b1,n1,b2,alpha) for b2 in np.arange(-1,1,.05)])*pn1(n1,b1)
    return p

betas = np.arange(-1,1,.05)



data =[]
labels=[]
labels_all=[]
for b1 in betas:
    for n1 in [0,1]:
        label1 = Q1_star(b1)
        for b2 in betas:
            label2 = Q2(b1,n1,b2)
            labels_all.append([label2])
            data.append([[b1,-1.],[b2,n1]])
            labels.append([label1, label2])

data = np.array(data).astype(np.float32)
labels=np.array(labels).astype(np.float32)

name="task2_dset"

if not os.path.exists(name):
    os.makedirs(name)

np.save(name+"/data",np.array(data), allow_pickle=True)
np.save(name+"/label",np.array(labels), allow_pickle=True)


    # p0 = pn1(0,b1)
    # n1 = np.random.choice([0,1],p=[p0,1-p0])
