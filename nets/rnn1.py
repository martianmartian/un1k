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

def clip(g,limit=1.5):
    # mi=-limit
    # ma=limit
    # absg=np.abs(g)
    # if np.max(absg)>limit:
    #     g=np.clip(g,mi,ma)
    g=np.tanh(g)
    return g

def visualize(loss):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(loss)),loss,'r')
    plt.show()

def trainer(net,test=0,save=None,cycles=0,lr=0,decay=0,limit=0):
    # three tests:
    #     1. used when building nets the first time. 
    #             small data set, fast test
    #     2. used when testing basic learning ability
    #     3. a little more complicated testing

    if test==1:
        from nets.rnn1_.data0 import x,xindices,yindices
        net.learn(x=x,xindices=xindices,yindices=yindices)

    elif test==2:
        Loss=[]
        from nets.rnn1_.data1 import getter
        net.lr=lr
        t=0
        for cycle in range(cycles):
            print('cycle ',cycle)
            data=getter()
            for i in range(1000):
                x,xindices,yindices,word = next(data)
                net.learn(x=x,xindices=xindices,yindices=yindices,lr=lr,limit=0)

                if t>1200 and i%10==0 :
                    net.lr = net.lr*decay
                
                if i % 10 ==0:
                    t+=1
                    Loss.append(net.loss(yindices))

                    # saveto=save+str(cycle)+'.'+str(i)
                    # np.save(saveto+'.loss.npy',Loss)
                    # np.save(saveto+'.lr.npy',net.lr)
                    # np.save(saveto+'.W.npy',net.W)
                    # np.save(saveto+'.U.npy',net.U)
                    # np.save(saveto+'.V.npy',net.V)

                    # if len(Loss)>2 and Loss[-1] > 4:
                    if t>1800 and t%100==0:
                        print('net.lr==> ', net.lr,'--- Loss[-1]==> ',Loss[-1])
                        delta=net.o
                        delta[np.arange(net.T), yindices] -= 1
                        print('word ==> \n',word)
                        print('x ==> \n',x)
                        print('net.W ==> \n',net.W)
                        print('net.U ==> \n',net.U)
                        print('net.V ==> \n',net.V)
                        print('net.s ==> \n',net.s)
                        print('net.o ==> \n',net.o)
                        print('net.y_hats ==> \n',net.y_hats)
                        print('delta ==> \n',delta)
                        break
        visualize(Loss)

class rnn:
    def __init__(self):
        self.dim=30
        self.lr=np.array([0])

        self.W=np.identity(30)/10 # main input
        self.U=np.identity(30)/10 # recurrent part
        # self.U = np.fliplr(np.identity(30))

        self.V=np.identity(30)/10 # softmax weights

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
            s[t] = relu(self.W[:,xindices[t]] + self.U.dot(s[t-1]))
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


    def learn(self, x=None,xindices=None,yindices=None,y=None, lr=0.00001, test=None, limit=1.5):

        # Note: don't read these code to start to debug.
        # start with the equations instead.

        # this learning approach aligns with my math equations
        # quite a few variables don't have to be cached:
        # or at least only need the value of t-1
        # dsdW
        # dsdU
        # dEds

        self.forward(xindices)

        dEdW = np.zeros(self.W.shape)
        dEdU = np.zeros(self.U.shape)

        dEdV = np.zeros(self.V.shape)

        dy = np.negative(self.o)
        dy[np.arange(self.T), yindices] += 1

        dsdW = np.zeros((self.T+1,self.dim,self.dim))
        dsdU = np.zeros((self.T+1,self.dim,self.dim))

        dEds=np.zeros((self.T,self.dim))

        N_prime = drelu(self.s)


        for t in np.arange(self.T):
            dEdV += np.outer(dy[t],self.s[t])
            dEdV = clip(dEdV)

            dEds[t] = dy[t].dot(self.V)
            dsdW[t] = N_prime[t][:,None] * (x[t] + self.U*dsdW[t-1])
            dEdW += dEds[t][:,None] * dsdW[t]
            dEdW = clip(dEdW)

            dsdU[t] = N_prime[t][:,None] * (self.s[t-1] + self.U*dsdU[t-1])
            dEdU += dEds[t][:,None] * dsdU[t]
            dEdU = clip(dEdU)
            
            # print('np.max(dEdV),np.max(dEdW),np.max(dEdU):==> \n',np.max(dEdV),np.max(dEdW),np.max(dEdU))
            # break

        self.U += lr * dEdW
        self.V += lr * dEdU
        self.W += lr * dEdV







