# -*- coding: utf-8 -*-

import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal

class GaussianMultivariate:
    
    def __init__(self):
        self.gaussiannb = GaussianNB()
        
    def fit(self, X, y):
        self.gaussiannb.fit(X, y)
            
        self.prior = self.gaussiannb.class_prior_
        self.sigma = self.gaussiannb.sigma_
        self.theta = self.gaussiannb.theta_
        self.classes = self.gaussiannb.classes_
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return probas.argmax(axis=1)
    
    def predict_proba(self, X):

        post_total = []
        for i in range(X.shape[0]):
            post_classes = np.zeros(self.classes.shape[0])
            for j in range(len(self.classes)):                
                # cov=np.diag(self.sigma[j])            
                post = abs(multivariate_normal.pdf(X[i], mean=self.theta[j], 
                                                   cov=self.sigma[j], 
                                                   allow_singular=True))
                post_classes[j] = post
            if post_classes.sum()==0:
                post_classes = np.ones(self.classes.shape[0])
            post_total.append(np.divide(post_classes, post_classes.sum()))
          
        return np.array(post_total)
