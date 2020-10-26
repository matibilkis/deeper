import imageio
from glob import glob
import numpy as np


for n in [8,64,512, 1600]:
    images = []

    ids=[]
    for k in glob("net_{}/*.index".format(n)):
        k=k.replace("net_{}/".format(n),'')
        k=k.replace(".index",'')
        if int(k) not in ids:
            ids.append(int(k))
    stoppings = ids
    stoppings = np.sort(stoppings)

    filenames=[]
    for kk in stoppings:
        filenames.append("evs_{}/{}.png".format(n,kk))
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('clipped_movie_lmm_{}.gif'.format(n), images, **{"duration":.5})
