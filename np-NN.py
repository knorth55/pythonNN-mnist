#!/usr/bin/env python

import numpy as np
import sys
import mnist

class NNetwork:
    def __init__(self):
        self.mnist = mnist.MNIST('./data')
        [self.training_data, self.training_label] = self.mnist.load_training()
        [self.testing_data, self.testing_label] = self.mnist.load_testing()
        self.training_data = self.imgload(self.training_data)
        self.testing_data = self.imgload(self.testing_data)
        self.input_num = len(self.training_data[0])                                 # num of input layer
        self.middle_num = 50                                                        # num of middle layer 
        self.output_num = 10                                                        # num of output layer 
        self.nu = 0.1                                                               # 0.01 ~ 0.5
        self.backPropN = 50                                                         # backpropagation step times

    def run(self,img):
        self.input_output = np.append(img,1.0)
        self.input_output = np.transpose(self.input_output)
        self.middle_output = self.sigmoid(np.dot(self.w1,self.input_output))
        self.middle_output = np.append(self.middle_output,1.0)
        self.output_output = self.sigmoid(np.dot(self.w2,self.middle_output))

    def backPropagation(self):
        for step in range(0, self.backPropN):
            for (i,img) in zip(self.training_label,self.training_data): 
                output_ref = np.zeros((1,self.output_num),dtype=np.double)
                output_ref[0][int(i)] = 1.0
                self.run(img)
                output_error = (self.output_output - output_ref) * self.sigmoid_d(self.output_output)
                # middle_error
                middle_error = np.dot(output_error,self.w2) * self.sigmoid_d(self.middle_output)
                middle_error = np.resize(middle_error,(1,self.middle_num))
                self.w2 -= self.nu * np.dot(output_error.T,np.atleast_2d(self.middle_output)) 
                self.w1 -= self.nu * np.dot(middle_error.T,np.atleast_2d(self.input_output))
            self.identify()
            print "BackPropagation Step " + str(step+1) + " finished"
        np.savetxt("w1.txt",self.w1)
        np.savetxt("w2.txt",self.w2)
     
    def imgload(self,data):
        returnlist = []
        for img in data:
            x = []
            for num in img:
                x.append(num/255.0)
            returnlist.append(x)
        return np.array(returnlist,dtype=np.double)

    def resetW(self):
        self.w1 = np.random.uniform(-1,1,(self.middle_num,self.input_num+1))      # input -> middle
        self.w2 = np.random.uniform(-1,1,(self.output_num,self.middle_num+1))     # middle -> output

    def loadW(self):
        self.w1 = np.loadtxt("w1.txt",dtype=np.double)                              # input -> middle
        self.w2 = np.loadtxt("w2.txt",dtype=np.double)                              # middle -> output

    def identify(self):
        total = 0
        correct = 0
        for (i,img) in zip(self.testing_label,self.testing_data):
            self.run(img)
            max_value = 0.0
            id_ = 0
            for j in range(0,len(self.output_output)):
                if max_value < abs(self.output_output[j]):
                    max_value = abs(self.output_output[j])
                    id_ = j
            if (id_ == i):
                correct += 1
            total += 1
        print "accuracy " + str(float(correct)/float(total))

    def sigmoid(self,sum_input):
        return 1.0/(1.0+np.exp(-sum_input))
    
    def sigmoid_d(self,sum_input):
        return (1-sum_input)*sum_input

if __name__ == '__main__':
    NN = NNetwork()
    if len(sys.argv) != 2:
        print 'argv[1] is needed' 
        sys.exit(0)
    if sys.argv[1] == 'load':
        NN.loadW()
        NN.identify()
    elif sys.argv[1] == 'run':
        NN.resetW()
        NN.backPropagation()
        NN.identify()
    else:
        print '''
        Usage:
        python neuralNetwork.py run 
            run back propagation for neural network and save w1 and w2

        python neuralNetwork.py load
            load w1.txt and w2.txt file and run
        '''
        sys.exit(0)


