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
    mi=-limit
    ma=limit
    absg=np.abs(g)
    if np.max(absg)>limit:
        g=np.clip(g,mi,ma)
    return g

def visualize(loss):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(loss)),loss,'r')
    plt.show()

def trainer(net,test=0,save=None,cycles=0, decay=0,limit=0,reg=0):
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
        from nets.rnn1_.data1 import getter,i2ch
        t=0
        for cycle in range(cycles):
            print('cycle ',cycle)
            data=getter()

            for i in range(1000):
                x,xindices,yindices,singular,plural = next(data)
                net.learn(x=x,xindices=xindices,yindices=yindices,limit=limit,reg=reg)

                if t>2000 and i%10==0 :
                    net.lr = net.lr*decay
                
                t+=1
                if i % 100 ==0:
                    Loss.append(net.loss(yindices))

                    # saveto=save+str(cycle)+'.'+str(i)
                    # np.save(saveto+'.loss.npy',Loss)
                    # np.save(saveto+'.lr.npy',net.lr)
                    # np.save(saveto+'.W.npy',net.W)
                    # np.save(saveto+'.U.npy',net.U)
                    # np.save(saveto+'.V.npy',net.V)
                
                if i%300 == 0:
                    print('plur:==> ',plural)
                    l=net.predict()
                    print('pred:==> ',''.join([i2ch[index] for index in l]))
                    print('true:==> ',singular)
                    print('\n')

                # if len(Loss)>2 and Loss[-1] > 4:
                if t>1000 and Loss[-1]>3.5:
                    # net.W=clip(net.W)

                    print('net.lr==> ', net.lr,'--- Loss[-1]==> ',Loss[-1])
                    delta=net.o
                    delta[np.arange(net.T), yindices] -= 1
                    print('plural ==> \n',plural)
                    print('x ==> \n',x)
                    print('net.W ==> \n',net.W)
                    print('net.U ==> \n',net.U)
                    print('net.V ==> \n',net.V)
                    print('net.s ==> \n',net.s)
                    print('net.o ==> \n',net.o)
                    print('net.y_hats ==> \n',net.y_hats)
                    print('delta ==> \n',delta)

        visualize(Loss)

class rnn:
    def __init__(self,lr=0.001):
        self.dim=30
        self.lr=lr

        self.W=np.identity(30)/10 # main input
        self.U=np.identity(30)/10 # recurrent part
        # self.U = np.fliplr(np.identity(30))
        # self.U = np.random.rand(30,30)/10

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

    def predict(self):
        return np.argmax(self.o, axis=1)

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

        dEdV = np.zeros(self.V.shape)

        dy = np.negative(self.o)
        dy[np.arange(self.T), yindices] += 1

        dsdW = np.zeros((self.T+1,self.dim,self.dim))
        dsdU = np.zeros((self.T+1,self.dim,self.dim))

        dEds=np.zeros((self.T,self.dim))

        N_prime = drelu(self.s)


        for t in np.arange(self.T):
            dEdV += np.outer(dy[t],self.s[t])
            dEdV = np.tanh(dEdV)

            dEds[t] = dy[t].dot(self.V)
            dsdW[t] = N_prime[t][:,None] * (x[t] + self.U*dsdW[t-1])
            dEdW += dEds[t][:,None] * dsdW[t]
            dEdW = np.tanh(dEdW)

            dsdU[t] = N_prime[t][:,None] * (self.s[t-1] + self.U*dsdU[t-1])
            dEdU += dEds[t][:,None] * dsdU[t]
            dEdU = np.tanh(dEdU)
            
            # print('np.max(dEdV),np.max(dEdW),np.max(dEdU):==> \n',np.max(dEdV),np.max(dEdW),np.max(dEdU))
            # break

        # self.U += self.lr * dEdW
        # self.V += self.lr * dEdU
        # self.W += self.lr * dEdV

        self.U += self.lr * (dEdW-reg*dEdW)
        self.V += self.lr * (dEdU-reg*dEdU)
        self.W += self.lr * (dEdV-reg*dEdV)

        # self.W=clip(self.W,limit=2)
        # self.V=clip(self.V,limit=2)






