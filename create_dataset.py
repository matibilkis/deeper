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

def Q1(b1,alpha=0.4):
    p=0
    for n1 in [0,1]:
        p+= max([Q2(b1,n1,b2,alpha) for b2 in np.arange(-1,1,.05)])*pn1(n1,b1)
    return p/2

betas = np.arange(-1,1,.05)

data =[]
labels=[]
labels_all=[]
for b1 in betas:
    p0 = pn1(0,b1)
    n1 = np.random.choice([0,1],p=[p0,1-p0])
    for b2 in betas:
        label2 = Q2(b1,n1,b2)
        labels_all.append([label2])
        data.append([[b1,-1.],[b2,n1]])
        labels.append(label2)


labels=np.array(labels).astype(np.float32)
data_all = np.reshape(data, (len(data),2,2)).astype(np.float32)
labels_all = np.reshape(labels_all, (len(labels),1))

if not os.path.exists("ProfilesSet"):
    os.makedirs("ProfilesSet")

np.save("ProfilesSet/data",np.array(data_all), allow_pickle=True)
np.save("ProfilesSet/label",np.array(labels_all), allow_pickle=True)
