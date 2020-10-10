import numpy as np
import tensorflow as tf
from tqdm import tqdm
from nets import RNNC
from glob import glob
import os
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.01)

args = parser.parse_args()


net = RNNC(lrr=args.lr)

if not os.path.exists("logs"):
    os.makedirs("logs")
    os.makedirs("logs/scalars")
    os.makedirs("logs/scalars/testing")
    os.makedirs("logs/scalars/training")
    os.makedirs("nets")

if not os.path.exists("nets"):
    os.makedirs("nets")

nrun = str(len(glob("logs/scalars/training/*")))
logdir1 = "logs/scalars/training"
logdir2 = "logs/scalars/testing"

train_loss = tf.summary.create_file_writer(logdir1+"/"+nrun)
test_loss = tf.summary.create_file_writer(logdir2+"/"+nrun)


data_all, labels_all = np.load("ProfilesSet/data.npy"),  np.load("ProfilesSet/label.npy")
dat_set = tf.data.Dataset.from_tensor_slices((data_all,labels_all))
data_all_test=tf.convert_to_tensor(data_all)
labels_all_test = tf.convert_to_tensor(labels_all)
dat_set.shuffle(buffer_size=10**4)

l=[]
cuts = len(data_all)/args.bs
ind=0
for k in tqdm(range(10**5/cuts))):

    for k1,k2 in list(dat_set.batch(args.bs)):
        l.append(net.train_step(k1,k2))
        ind+=1
        with train_loss.as_default():
            tf.summary.scalar('train_loss', l[-1], step=ind)
        with test_loss.as_default():
            tf.summary.scalar('test_loss', tf.reduce_mean(tf.keras.losses.MSE(net(data_all_test),labels_all_test)), step=ind)

### why don't you work ?!!

net.save_weights("nets/"+str(nrun)+"/")
