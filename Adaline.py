#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 13:34:36 2018

@author: ujjwal
"""

import numpy as np

class Adaline(object):
    def __init__(self, learn_rate = 0.01, n_iter = 50):
        self.learn_rate = learn_rate
        self.n_iter = n_iter
    
    def net_input(self, X):
        return np.dot(X,self.w_[1:])+self.w_[0]
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            
            self.w_[1:] += self.learn_rate * X.T.dot(errors)
            self.w_[0] += self.learn_rate * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self
    
    def activation(self,X):
        return self.net_input(X)
    
    def predict(self, X):
        return np.where(self.activation(X)>=0.0, 1, -1)
