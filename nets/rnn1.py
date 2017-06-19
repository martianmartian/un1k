import numpy as np

# designed to translate infections back to their original spelling
# this is rnn 30*30 square identity recurrent net
# initialized with identity matrix

# inputs should be 1 hot chars, only 24 of them.
# rest of the space left for other purposes.
    # note that the w.x should be using the one_hot picker approach since 
    # x is always one_hot
# note: y is also one_hot.... it's always picking than multiplying

# rectifier for hidden activation


def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def relu(x):
    return np.maximum(0,x)

def drelu(x):
    return 1.*(x>0)

def clip(g,limit=1):
    mi=-limit
    ma=limit
    absg=np.abs(g)
    if np.max(absg)>limit:
        g=np.clip(g,mi,ma)
    return g

class rnn:
    def __init__(self,lr=0.001):
        self.dim=30
        self.lr=lr

        self.W=np.identity(30)*0.5 # main input
        self.U=np.identity(30)*0.5 # recurrent part
        self.b=np.zeros(30) #bias

        self.V=np.identity(30)*0.01 # softmax weights

        self.y_hats=None
        self.s=None
        self.o=None
        self.T=None  # steps in each learning update cycle


    def forward(self,xindices):
        # The total number of time steps
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        # The outputs at each time step. Again, we save them for later.
        # For each time step...
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            # picker s[t] = relu(self.U[:,x[t]] + self.W.dot(s[t-1])) 
        # if x not defined, 
            # use default fake data for testing

        T = len(xindices)
        s = np.zeros((T + 1, self.dim))
        o = np.zeros((T, self.dim))
        for t in np.arange(T):
            s[t] = relu(self.W[:,xindices[t]] + self.U.dot(s[t-1]) + self.b)
            o[t] = softmax(self.V.dot(s[t]))

        self.o=o
        self.s=s
        self.T=T


    def loss(self,yindices=None):
        
        Loss=0

        self.y_hats = self.o[np.arange(self.T), yindices]
        Loss += -1 * np.sum(np.log(self.y_hats))
        Loss /= self.T

        return Loss


    def learn(self, x=None,xindices=None,yindices=None,y=None, test=None, limit=1,reg=0):

        # Note: don't read these code to start to debug.
        # start with the equations instead.

        # this learning approach aligns with my math equations
        # quite a few variables don't have to be cached:
        # or at least only need the value of t-1
        # dsdW
        # dsdU
        # dEds

        # here, np.tanh is used to clip the gradients
            # surprisingly it works pretty well
        # -0.1* is used to do regularization

        self.forward(xindices)

        dEdW = np.zeros(self.W.shape)
        dEdU = np.zeros(self.U.shape)
        dEdb = np.zeros(self.b.shape)

        dEdV = np.zeros(self.V.shape)

        dy = np.negative(self.o)
        dy[np.arange(self.T), yindices] += 1

        dsdW = np.zeros((self.T+1,self.dim,self.dim))
        dsdU = np.zeros((self.T+1,self.dim,self.dim))
        dsdb = np.zeros((self.T+1,self.dim))

        dEds=np.zeros((self.T,self.dim))

        N_prime = drelu(self.s)


        for t in np.arange(self.T):

            dEdV += np.outer(dy[t],self.s[t])

            dEds[t] = dy[t].dot(self.V)

            cache=self.U.dot(dsdW[t-1])
            cache[:,xindices[t]]+=1
            dsdW[t] = N_prime[t][:,None] * cache
            dEdW += dEds[t][:,None] * dsdW[t]

            dsdU[t] = N_prime[t][:,None] * (self.s[t-1] + self.U.dot(dsdU[t-1]))
            dEdU += dEds[t][:,None] * dsdU[t]
            
            dsdb[t] = (N_prime[t][:,None] * (self.U * dsdb[t-1][:,None])).mean(axis=1)
            dEdb += dEds[t] * dsdb[t]

            # print('np.max(dEdV),np.max(dEdW),np.max(dEdU):==> \n',np.max(dEdV),np.max(dEdW),np.max(dEdU))

        dEdW = np.tanh(dEdW)
        dEdU = np.tanh(dEdU)
        dEdb = np.tanh(dEdb)   # clip inside or outside

        dEdV = np.tanh(dEdV)

        self.W += self.lr * dEdW
        self.U += self.lr * dEdU
        self.b += self.lr * dEdb

        self.V += self.lr * dEdV




