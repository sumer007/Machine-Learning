'''
Created on 29-Aug-2015
@author: SUMER
'''

from __future__ import division
import os
import numpy as np
from tkFileDialog import askopenfilename

os.chdir("C:\Users\SUMER\Desktop\Sumer\GRAPHICS\Python\ml")

class ANN:
    """
    Artificial Neural Network
    """
    def __init__(self, x, y, num_hlayers=1, num_units=5, num_iters=1000, alpha=0.01, l=0):
        self.x = np.array(x).T
        self.y = np.vstack(y).T
        self.num_datas = len(y)
        self.num_hlayers = num_hlayers
        self.num_units = num_units
        self.num_iters = num_iters
        self.alpha = alpha
        self.l = l
        self.theta = []
        self.net = [len(self.x)]
        for i in range(num_hlayers):
            self.net.append(num_units)
        self.net.append(len(self.y))
        for i in range(num_hlayers+1):
            self.theta.append(np.ones((self.net[i+1],self.net[i])))
        self.layers = []
        self.layers.append(self.x)
        for i in range(self.num_hlayers):
            self.layers.append(np.ones((self.num_units,self.num_datas)))
        self.layers.append(self.y)
            
        # degugging
        #print "x shape = ",self.x.shape
        #print "y shape = ",self.y.shape
        #print "num of datas = ",self.num_datas
        #print "theta = ",self.theta
        #for i in range(len(self.layers)):
            #print "layers", i, "is",self.layers[i].shape
        
    def sig_fn(self,z):
        return 1 / (1 + np.exp(z))
    
    def tanh_fn(self,z):
        return np.tanh(z)
    
    def run_ann(self):
        it = 0
        while it < self.num_iters:
            it += 1
            for layer in range(self.num_hlayers+1):
                self.layers[layer+1] = self.g()
        return
        

if __name__ == "__main__":
    f = askopenfilename()
    data = np.loadtxt(f)
    x = data[:,:-1]
    y = data[:,-1]
    
    ann = ANN(x,y)
    ann.run_ann()
    