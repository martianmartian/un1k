# a simple test to see if rnn learns anything at all:
#     plural of words.
#         this test gets arbituary valid words
#         use plurals as inputs
#         use singluars as outputs
#         for example:
#             dogs dog_
#             cats cat_
#             boys boy_
#             womans woman_
#             mans man_
#             horses horse_
#         note that some aren't real. just use it to .... test
#         note that eventually 
#             we only need the first 1000 words' 
#             we need passtent, and other forms as well. different forms


import numpy as np
from components.w2v import i2w

plurals=[w+'s' for w in i2w]
singluars=[w+'_' for w in i2w]
# plurals=[w for w in i2w]
# singluars=[w for w in i2w]


i2ch=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','_',]
ch2i=dict((j,i) for i,j in enumerate(i2ch))

def getter():
    for i in range(len(i2w)):
        xindices=[ch2i[ch.lower()] for ch in plurals[i]]
        yindices=[ch2i[ch.lower()] for ch in singluars[i]]
        T=len(xindices)
        x=np.zeros((T,30))
        x[np.arange(T),xindices]=1

        yield [x,xindices,yindices,singluars[i]]



