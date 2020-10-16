import imageio
from glob import glob
images = []


stoppings = []
for k in range(4):
    for kk in range(10**k,10**(k+1),10**k):
        stoppings.append(kk)
filenames=[]
for k in stoppings:
    filenames.append("giffy_clip/{}.png".format(k))
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('clipped_movie.gif', images, **{"duration":1})
