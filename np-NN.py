#!/usr/bin/env python

import numpy as np
import sys
import random
import mnist
import cv2

class NNetwork:
    def __init__(self):
        self.mnist = mnist.MNIST('./data')
        self.middle_num = 10                                                        # num of middle layer 
        self.output_num = 10                                                        # num of output layer 
        self.nu = 0.1                                                               # 0.01 ~ 0.5
        self.backPropN = 30                                                         # backpropagation step times

    def load_training_data(self):
        print "start loading training data"
        [self.training_data, self.training_label] = self.mnist.load_training()
        self.training_data = self.imgload(self.training_data,1)
        self.input_num = len(self.training_data[0])                                 # num of input layer
        print "finish loading training data"
    
    def load_tesing_data(self): 
        print "start loading testing data"
        [self.testing_data, self.testing_label] = self.mnist.load_testing()
        self.testing_data = self.imgload(self.testing_data,0)
        print "finish loading testing data"
 
    def run(self,img):
        self.input_output = np.append(img,1.0)
        self.input_output = np.transpose(self.input_output)
        self.middle_output = self.sigmoid(np.dot(self.w1,self.input_output))
        self.middle_output = np.append(self.middle_output,1.0)
        self.output_output = self.sigmoid(np.dot(self.w2,self.middle_output))

    def backPropagation(self):
        print "start back propagation"
        accuracy_prev = 0.0
        for step in range(0, self.backPropN):
            print "------------------------------"
            for (i,img) in zip(self.training_label,self.training_data): 
                output_ref = np.zeros((1,self.output_num),dtype=np.double)
                output_ref[0][int(i)] = 1.0
                self.run(img)
                # output error
                output_error = (self.output_output - output_ref) * self.sigmoid_d(self.output_output)
                # middle_error
                middle_error = np.dot(output_error,self.w2) * self.sigmoid_d(self.middle_output)
                middle_error = np.resize(middle_error,(1,self.middle_num))
                # w2 update
                self.w2 -= self.nu * np.dot(output_error.T,np.atleast_2d(self.middle_output)) 
                # w1 update
                self.w1 -= self.nu * np.dot(middle_error.T,np.atleast_2d(self.input_output))
            self.identify()
            if (self.accuracy < accuracy_prev):
                print "Warning: Accuracy is Decreasing !!"
            accuracy_prev = self.accuracy
            print "BackPropagation Step " + str(step+1) + " finished"
            print "------------------------------"
        np.savetxt("w1.txt",self.w1)
        np.savetxt("w2.txt",self.w2)
        print "w1 and w2 saved and back propagation finished"

    def imgload(self,data,noise_flag):
        if len(sys.argv) == 3 and noise_flag == 1:
            if float(sys.argv[2]) > 1.0:
                print "noise rate should be < 1.0"
                sys.exit(0)
            noise = float(sys.argv[2])
            print "noise rate: " + str(noise)
        else:
            noise = 0.0
        returnlist = []
        for img in data:
            x = []
            for num in img:
                if random.random() < noise:
                    num = random.random()
                else:
                    num = num /255.0
                x.append(num)
            returnlist.append(x)
        return np.array(returnlist,dtype=np.double)

    def initW(self):
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
        self.accuracy = float(correct)/float(total) 
        print "accuracy: " + str(self.accuracy)

    def optimal_img(self):
        dirpath = "img/"
        i = 0
        for img in self.w1:
            img = np.resize(img,(1,len(img)-1))
            img = np.reshape(img,(28,28)) * 255
            filepath = dirpath + "w1-" + str(i) + ".png"
            cv2.imwrite(filepath,img)
            i = i+1
    
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    
    def sigmoid_d(self,x):
        return (1-x)*x

if __name__ == '__main__':
    NN = NNetwork()
    if len(sys.argv) < 2:
        print 'argv[1] is needed' 
        sys.exit(0)
    if sys.argv[1] == 'load':
        NN.load_tesing_data()
        NN.loadW()
        NN.identify()
    elif sys.argv[1] == 'run':
        NN.load_training_data()
        NN.load_tesing_data()
        NN.initW()
        NN.backPropagation()
        NN.identify()
    elif sys.argv[1] == 'optimal':
        NN.loadW()
        NN.optimal_img()
    else:
        print '''
        Usage:
        python np-NN.py run [noise_rate] 
            run back propagation for neural network and save w1 and w2
            option: noise rate for training data (default = 0.0)

        python np-NN.py load
            load w1.txt and w2.txt file and run

        python np-NN.py optimal
            load w1.txt and create optimal stimulation imgs in /img
        '''
        sys.exit(0)


