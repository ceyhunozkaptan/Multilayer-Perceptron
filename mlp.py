# %%
import numpy as np
import matplotlib.pyplot as plt
from random import random
from random import seed
from math import exp

# %% Neuron Class to build the network
class Neuron:
    output = 0          # y
    localGradient = 0   # delta

    # Initializer / Instance Attributes
    def __init__(self, nInputs):
        self.weights = [0] * (nInputs+1)        # w
        self.prevUpdate = [0] * (nInputs+1)     # delta w -> for momentum

        self.weights[-1] = 2*random()-1         # the last weight is the bias
        for i in range(nInputs):
            self.weights[i] = 2*random()-1

    def response(self, input):                  # Output function
        activation = self.weights[-1]
        for i in range( len(self.weights)-1 ):
            activation += self.weights[i] * input[i]
        self.output = np.tanh(activation)             # Tanh()
        # self.output = (1.0 / (1.0 + exp(-activation)))  # Sigmoid 
        return self.output

# %% Forward Propagation Algorithm
def forward_computation(neuralNet, x):
    temp = x

    for layer in neuralNet:
        outputs = []
        for node in layer:      
            outputs.append( node.response(temp) )
        temp = outputs
    return outputs

# %% Backward Propagation Algorithm
def backward_computation(neuralNet, x, desired, alpha, eta):
    for l in reversed(range(len(neuralNet))):
        layer = neuralNet[l]

        if (l == len(neuralNet)-1):     # Output layer (l: layer, j: neuron, i: weight)
            outNeuron = layer[0]
            
            # Error and Local Dradient
            error = desired - outNeuron.output
            outNeuron.localGradient =  error * (1/np.cosh(outNeuron.output)**2)               # Tanh derivative
            # outNeuron.localGradient =  error * (outNeuron.output * (1.0 - outNeuron.output))    # Sigmoid derivative
            
            # Weight Updates
            for i in range(len(outNeuron.weights)):
                if (i == len(outNeuron.weights)-1): # Bias Update
                    weightUpdate = alpha*outNeuron.prevUpdate[i] + eta*outNeuron.localGradient
                else:
                    weightUpdate = alpha*outNeuron.prevUpdate[i] + eta*outNeuron.localGradient*neuralNet[l-1][i].output

                outNeuron.weights[i] = outNeuron.weights[i] + weightUpdate
                outNeuron.prevUpdate[i] = weightUpdate

        else:                           # Hidden Layers (l: layer, j: neuron, i: weight)
            for j in range(len(layer)):
                neuron = layer[j]
                
                # Propagated Error and Local Gradient
                error = neuralNet[-1][0].weights[j] * neuralNet[-1][0].localGradient
                neuron.localGradient =  error * (1/np.cosh(neuron.output)**2)             # Tanh derivative
                # neuron.localGradient =  error * (neuron.output * (1.0 - neuron.output))   # Sigmoid derivative

                # Weight Updates
                for i in range(len(neuron.weights)-1):
                    if (i == len(neuron.weights)): # Bias Update
                        weightUpdate = alpha*neuron.prevUpdate[i] + eta*neuron.localGradient
                    else:
                        weightUpdate = alpha*neuron.prevUpdate[i] + eta*neuron.localGradient*x[i]
                    neuron.weights[i] = neuron.weights[i] + weightUpdate
                    neuron.prevUpdate[i] = weightUpdate

# %% Generating the training set
nInput = 5
xTrain = []
yTrain = [0] * (2**nInput)

temp = [0] * nInput
for i in range(2**nInput):
    binaryStr = f'{i:05b}'

    if (binaryStr.count('1') % 2 == 0):
        # yTrain[i] = 0   # For Sigmoid activation
        yTrain[i] = -1  # For Tanh activation
    else:
        yTrain[i] = 1

    xTrain.append([0]*nInput)
    for n in range(nInput):
        if ( int(binaryStr[n]) == 1 ):
            xTrain[i][n] = 1
        else:
            xTrain[i][n] = -1

# %% Training for alpha = 0
nHiddenNodes = 8

# etaValues = np.arange(0.05, 0.5, 0.05)   # For Sigmoid activation
etaValues = np.arange(0.005, 0.05, 0.005)   # For Tanh activation

maxEpoch = 10000
results1 = [maxEpoch] * len(etaValues)

alpha = 0

it = 0
for eta in etaValues:

    # Building the neural network using Neuron class
    neuralNet = []

    layer = []
    for i in range(nHiddenNodes):
        layer.append(Neuron(nInput))
    neuralNet.append(layer)

    outLayer = []
    outLayer.append(Neuron(nHiddenNodes))
    neuralNet.append(outLayer)

    # Training and Results
    for n in range(maxEpoch):
        mse = 0.0
        absError = 0.0
        for i in range(len(xTrain)):
            x = xTrain[i]
            d = yTrain[i]
            y = forward_computation(neuralNet, x)
            backward_computation(neuralNet, x, d, alpha, eta)
            
            mse += (d - y[0])**2
            absError = max(abs(d - y[0]), absError)

        mse = mse/32
        if (n % 100 == 0):
            print ('Epoch %d > eta = %.3f, MSE = %.3f, AE = %.3f' % (n,eta,mse,absError))

        if absError <= 0.1:
            print ('Epoch %d > eta = %.3f, MSE = %.3f, AE = %.3f' % (n,eta, mse,absError))
            results1[it] = n
            break
    it += 1

# %% Training for alpha = 0.8
results2 = [maxEpoch] * len(etaValues)

alpha = 0.8

it = 0
for eta in etaValues:
    # Building the neural network using Neuron class
    neuralNet = []

    layer = []
    for i in range(nHiddenNodes):
        layer.append(Neuron(nInput))
    neuralNet.append(layer)

    outLayer = []
    outLayer.append(Neuron(nHiddenNodes))
    neuralNet.append(outLayer)

    # Training and Results
    for n in range(maxEpoch):
        mse = 0.0
        absError = 0.0
        for i in range(len(xTrain)):
            x = xTrain[i]
            d = yTrain[i]
            y = forward_computation(neuralNet, x)
            backward_computation(neuralNet, x, d, alpha, eta)
            
            mse += (d - y[0])**2
            absError = max(abs(d - y[0]), absError)

        mse = mse/32
        if (n % 100 == 0):
            print ('Epoch %d > eta = %.3f, MSE = %.3f, AE = %.3f' % (n,eta,mse,absError))

        if absError <= 0.1:
            print ('Epoch %d > eta = %.3f, MSE = %.3f, AE = %.3f' % (n,eta, mse,absError))
            results2[it] = n
            break
    it += 1

# %% Plotting Results
plt.figure(figsize=(8,5))
plt.plot(etaValues, results1, label=r'$\alpha = 0$', marker='o', fillstyle='none')
plt.plot(etaValues, results2, label=r'$\alpha = 0.8$', marker='s', fillstyle='none')
plt.xlabel(r"$\eta$")
plt.ylabel('Number of Epochs')
plt.title(r'Activation Function: Tanh')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':')
plt.show()
