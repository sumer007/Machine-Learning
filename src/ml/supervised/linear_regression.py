from __future__ import division
import os
import numpy as np
import shelve
import xlrd
from tkFileDialog import askopenfilename

#print os.getcwd()
os.chdir("C:\Users\SUMER\Desktop\Sumer\GRAPHICS\Python\ml")

########################### Multivariate Multiple Output Linear Regression ############################

class MLR:
    """ Multivariate Linear Regression:
    x = input matrix with row as features
    y = output matrix
    theta = initial value of the parameters
    num_iters = number of iterations
    aplha = learning rate
    l = regularization parameter
    """
    def __init__(self, x, y, theta=None, num_iters=1000, alpha=0.01, l=0, save_name=None):                          # constructor
        self.x = np.array(x)
        self.y = np.array(y)
        self.num_outputs = len(self.y[0])
        self.num_datas = len(self.y)
        #self._normalise_inputs()
        self.num_features = len(self.x[0])
        if theta is None:
            self.theta = 10 * np.random.rand(self.num_features,self.num_outputs)
        else:
            self.theta = theta
        self.theta = np.array(self.theta)
        print "theta generated = ",self.theta
        self.num_iters = num_iters
        self.alpha = alpha
        self.l = l
        self.save_name = save_name
        self.cost_vals = []
        self.xmean = []
        self.xstd = []
        
    def _normalise_inputs(self):
        self.xmean = self.x.mean(axis=0)
        self.xstd = self.x.std(axis=0)
        self.x = (self.x - self.xmean)/self.xstd
        ones_mat = np.ones((self.num_datas,1))
        self.x = np.concatenate((ones_mat,self.x),axis=1)
        
    def _h(self, theta):
        #print "hypothesis = ",np.dot(self.x, theta)
        return np.dot(self.x, theta)
        
    def _cost(self, theta):
        h = self._h(theta)
        cost = sum((h-self.y)**2) + self.l * sum(sum(theta[1:]))
        return sum(cost)/len(cost)
    
    def _gradient_descent(self):
        it = 0
        while it < self.num_iters:
            print "iter = ",it
            a = (1 - self.alpha * self.l / self.num_datas)
            #print "a = ",a
            self.theta = a * self.theta - self.alpha / self.num_datas * np.dot(self.x.T, self._h(self.theta) - self.y)
            #self.cost_vals.append(self._cost(self.theta))
            it += 1
            print "theta = ",self.theta
        return self.theta, self.cost_vals
    
    def run_mlr(self):
        print "running mlr"
        opt_vals = self._gradient_descent()
        if not self.save_name is None:
            self._preserve_data()
        return opt_vals
    
    def _preserve_data(self):
        self.save_name = self.save_name + ".dat"
        data_file = shelve.open(self.save_name)
        data_file["theta"] = self.theta
        data_file["xmean"] = self.xmean
        data_file["xstd"] = self.xstd
        data_file.sync()
        data_file.close()
        print "data preserved"
        
if __name__ == '__main__':
    f = askopenfilename()
    ext = os.path.splitext(f)[1]
    if ext == ".xls" or ext == ".xlsx":
        book = xlrd.open_workbook(f)
        sheet = book.sheet_by_index(0)
        x = np.vstack(sheet.col_values(0))
        y = np.vstack(sheet.col_values(1))
    elif ext == ".txt":
        data = np.loadtxt(f)
        x = np.array(data[:,:-1])
        y = np.vstack(data[:,-1])
        print "x = ",len(x)
        print "y = ",len(y)
    #print "x = ",x
    #print "y = ",y
    mlr = MLR(x[:200],y[:200])
    opt_vals = mlr.run_mlr()
    theta = opt_vals[0]
    print "theta = ",theta