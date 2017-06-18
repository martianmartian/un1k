import numpy as np

# this is designed to continue after the rnn1
#     racees was converted to rac_e_
# in order to combat problems of it's kind, new structure is created here in rnn2


def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def relu(x):
    return np.maximum(0,x)

def drelu(x):
    return 1.*(x>0)


class rnn:
    def __init__(self,lr=0.001):
        self.dim=30
        self.span=5
        self.lr=lr

        self.W=np.tile(np.identity(self.dim)*0.1,(1,self.span))
        self.U=np.identity(self.dim)*0.8
        self.b=np.zeros(30) #bias

        self.V=np.identity(self.dim)*0.01           # softmax weights

        self.y_hats=None
        self.s=None
        self.o=None
        self.T=None  # steps in each learning update cycle


    def forward(self,xindices,yindices):

        T = len(yindices)
        s = np.zeros((T + 1, self.dim))
        o = np.zeros((T, self.dim))

        for t in np.arange(T):
            h=0
            for i in np.arange(self.span):
                h+=self.W[:,self.dim*i+xindices[t+1]]
            s[t]+=relu(h+self.U.dot(s[t-1])+self.b)
            o[t] = softmax(self.V.dot(s[t]))

        self.o=o
        self.s=s
        self.T=T


    def learn(self, x=None,xindices=None,yindices=None,y=None, test=None, limit=1,reg=0):

        self.forward(xindices,yindices)

        dEdW = np.zeros(self.W.shape)
        dEdU = np.zeros(self.U.shape)
        dEdb = np.zeros(self.b.shape)

        dEdV = np.zeros(self.V.shape)

        dy = np.negative(self.o)
        dy[np.arange(self.T), yindices] += 1

        dsdW = np.zeros((self.T+1,self.dim,self.dim * self.span))
        dsdU = np.zeros((self.T+1,self.dim,self.dim))
        dsdb = np.zeros((self.T+1,self.dim))

        dEds=np.zeros((self.T,self.dim))

        N_prime = drelu(self.s)


        for t in np.arange(self.T):

            dEdV += np.outer(dy[t],self.s[t])

            dEds[t] = dy[t].dot(self.V)

            # for i in np.arange(self.span):
            #     # h+=self.W[:,self.dim*i+xindices[t+1]]
            print((self.U*dsdW[t-1]).shape)
            
            dsdW[t] = N_prime[t][:,None] * (x[t] + self.U*dsdW[t-1])
            dEdW += dEds[t][:,None] * dsdW[t]

            # dsdU[t] = N_prime[t][:,None] * (self.s[t-1] + self.U*dsdU[t-1])
            # dEdU += dEds[t][:,None] * dsdU[t]
            
        #     dsdb[t] = (N_prime[t][:,None] * (self.U * dsdb[t-1][:,None])).mean(axis=1)
        #     dEdb += dEds[t] * dsdb[t]

        #     # print('np.max(dEdV),np.max(dEdW),np.max(dEdU):==> \n',np.max(dEdV),np.max(dEdW),np.max(dEdU))

        # dEdV = np.tanh(dEdV)
        # dEdW = np.tanh(dEdW)
        # dEdU = np.tanh(dEdU)
        # dEdb = np.tanh(dEdb)   # clip inside or outside

        # self.W += self.lr * (dEdV-reg*dEdV)
        # self.U += self.lr * (dEdW-reg*dEdW)
        # self.b += self.lr * (dEdb-reg*dEdb)

        # self.V += self.lr * (dEdU-reg*dEdU)



    def loss(self,yindices=None):
        
        Loss=0

        return Loss
































