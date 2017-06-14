import numpy as np

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
        Accu=[]
        t=0
        from nets.rnn1_.data1 import getter,i2ch,get_accu
        for cycle in range(cycles):
            print('cycle ',cycle)
            data=getter()

            for i in range(1000):
                t+=1
                x,xindices,yindices,singular,plural = next(data)
                net.learn(x=x,xindices=xindices,yindices=yindices,limit=limit,reg=reg)

                if t>2000 and i%10==0 :
                    net.lr = net.lr*decay

                if t>1000 and Loss[-1]>3.5:
                    # this part prints the weights if loss starts to go up
                    print('\nnet.lr==> ', net.lr,'--- Loss[-1]==> ',Loss[-1])
                    delta=net.o
                    delta[np.arange(net.T), yindices] -= 1
                    print('plural ==> \n',plural)
                    print('x ==> \n',x)
                    print('net.W ==> \n',net.W[:8,:8])
                    print('net.U ==> \n',net.U[:8,:8])
                    print('net.V ==> \n',net.V[:8,:8])
                    print('net.s ==> \n',net.s[:8,:8])
                    print('net.o ==> \n',net.o[:8,:8])
                    print('net.y_hats ==> \n',net.y_hats)
                    print('delta ==> \n',delta[:8,:8])
                
                if i % 100 ==0:
                    # this part visualizes loss and accuracy
                    Loss.append(net.loss(yindices))
                    data1=get_accu()
                    pred=[]
                    true=[]
                    for i in range(94):
                        x,xindices,yindices,singular,plural = next(data1)
                        net.forward(xindices)
                        l=np.argmax(net.o, axis=1)
                        pred.append(l)
                        true.append(yindices)
                    
                    Accu.append(np.equal(np.concatenate(pred),np.concatenate(true)).mean())
                    
                    # print('current plur:==> ',plural)
                    # l=np.argmax(net.o, axis=1)
                    # print('current pred:==> ',''.join([i2ch[index] for index in l]))
                    # print('current true:==> ',singular)
                    # print('\n')


                    # saveto=save+str(cycle)+'.'+str(i)
                    # np.save(saveto+'.loss.npy',Loss)
                    # np.save(saveto+'.lr.npy',net.lr)
                    # np.save(saveto+'.W.npy',net.W)
                    # np.save(saveto+'.U.npy',net.U)
                    # np.save(saveto+'.V.npy',net.V)
                

        visualize(Loss)
        visualize(Accu)
