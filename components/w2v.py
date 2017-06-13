import numpy as np


from components.dicts1000 import i2w
w2i=dict((j,i) for i,j in enumerate(i2w))
U_embed=np.zeros((1000,1000))
W_identity=np.eye(1000,dtype=np.int)
# print("once")
if False:
    print("w2i['what']: ",w2i['what'])
    print("i2w[32]: ",i2w[32])
    print(U_embed[w2i['what']][:35])
    print(W_identity[w2i['what']][:35])

