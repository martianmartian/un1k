import numpy as np
import sys

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=10, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x, vobose=False):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim) # last state is actually the first state when processing......
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        if vobose: print('\no.shape is: ',o.shape)
        if vobose: print('output is: \n',o[0:3,0:3])
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            supposed_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(supposed_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])              
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    #BPTT: Using only one loop; from github
    def bptt_2(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        
        dsprev = np.zeros(s[0].shape)

        # For each output backwards...
        for t in np.arange(T)[::-1]:

            dLdV += np.outer(delta_o[t], s[t].T)
            dst = self.V.T.dot(delta_o[t])
            
            #Add previous ds to current one
            dst += dsprev
            dtanh = dst * (1 - (s[t] ** 2)) # 1 - (output_of_tanh^2)

            dLdW += np.outer(dtanh, s[t-1].T)
            x_one_hot = np.zeros(o[0].shape)
            x_one_hot[x[t]] = 1
            dLdU += np.outer(dtanh, x_one_hot)
            dsprev = self.W.T.dot(dtanh)

        return [dLdU, dLdV, dLdW]

    # Performs one step of SGD.
    def numpy_sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate;
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    # Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
    def train_with_sgd(self, X_train, y_train,verbose=False, learning_rate=0.01, nepoch=100, evaluate_loss_after=5):
        import sys
        from datetime import datetime
        print("\n\ntraining beging: =======> ")
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if verbose: print( "\n%s: Loss after %d_examples @epoch=%d: %f lr=%f" % (time, num_examples_seen, epoch, loss,learning_rate))
                # Adjust the learning rate if loss increases
                if verbose: print(losses)
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    if verbose: print( "Setting learning rate to %f" % learning_rate)
                elif(len(losses) > 1 and losses[-1][1] < losses[-2][1]):
                    learning_rate = learning_rate + 0.001

                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                self.numpy_sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1

    def generate_sentence(self,w_index,word_to_index,index_to_word,verbose=False):
        print("\ngenerating sentences: =======> ")
        if verbose: print([index_to_word[i] for i in w_index])
        wout = self.forward_propagation(w_index)
        if verbose: print(np.array(wout).shape)
        if verbose: print(np.array(wout[0]).shape)
        if verbose: print(wout[0:1])
        wlist = [index_to_word[i] for i in [np.argmax(w[0]) for w in wout]]
        return wlist

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print ("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print ("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print ("+h Loss: %f" % gradplus)
                    print ("-h Loss: %f" % gradminus)
                    print ("Estimated_gradient: %f" % estimated_gradient)
                    print ("Backpropagation gradient: %f" % backprop_gradient)
                    print ("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            print ("Gradient check for parameter %s passed." % (pname))
 
