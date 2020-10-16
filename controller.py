import os
import multiprocessing as mp

def fun(v):
    lr, bs = v
    os.system("python3 train.py --lr "+str(lr) + " --bs "+str(bs)+ " --its "+str(10**4))



vals=[]
for lr in [0.01]:
    for bs in [8, 64, 128]:
        vals.append([lr,bs])
with mp.Pool(4) as p:
    p.map(fun, vals)
