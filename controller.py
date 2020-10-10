import os

for lr in [0.01, 0.001]:
    for bs in [8, 64, 128, 512]:
        os.system("python3 train.py --lr "+str(lr) + " --bs "+str(bs))
