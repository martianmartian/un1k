import numpy as np


# this file creates some arbituary data for testig the net as a feedback net
 
np.random.seed(0)
# x=np.random.rand(5,30)
x=np.zeros((5,30))
xindices=np.random.randint(30,size=5)
x[np.arange(5),xindices]=1

y=np.zeros((5,30))
yindices=np.random.randint(30,size=5)
y[np.arange(5),yindices]=1

