import imageio
from glob import glob


images = []

for n in [1,2]:

    stoppings = []
    for k in range(4):
        for kk in range(10**k,10**(k+1),10**k):
            stoppings.append(kk)
    filenames=[]
    for k in stoppings:
        filenames.append("evs_{}/{}.png".format(n,k))
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('clipped_movie_lmm_{}.gif'.format(n), images, **{"duration":1})
