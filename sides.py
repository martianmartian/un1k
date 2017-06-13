import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,0,0,2,0,1])
def softmax(x):
    exp = np.exp(x)
    return exp/np.sum(exp)

y=softmax(x)
print(y)
# xx=np.arange(5)
# plt.bar(x, y,label='y')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
