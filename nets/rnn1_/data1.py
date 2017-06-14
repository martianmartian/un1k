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

plurals=[w+'es' for w in i2w]
singluars=[w+'__' for w in i2w]


i2ch=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','_',]
ch2i=dict((j,i) for i,j in enumerate(i2ch))

def convert(i):
    xindices=[ch2i[ch.lower()] for ch in plurals[i]]
    yindices=[ch2i[ch.lower()] for ch in singluars[i]]
    T=len(xindices)
    x=np.zeros((T,30))
    x[np.arange(T),xindices]=0.1
    return [x,xindices,yindices,singluars[i],plurals[i]]

def getter():
    for i in range(len(i2w)):
        yield convert(i)


accuIndices = np.array([ 3, 22,42,58,66,70, 74,82,83,88,93, 122, 140,165,166,170,187,192,96,202,205,208,213,214,225,245,249,305,316,349,377,95,396,399,411,428,441,442,443,451,470,473,478,500,17,525,533,545,548,562,581,588,596,604,616,618,623,42,643,658,664,680,695,706,707,715,718,732,747,749,50,754,762,798,800,804,811,839,869,881,890,903,924,25,941,945,948,954,956,963,965,968,980, 997])
def get_accu():
    for i in accuIndices:
        yield convert(i)


