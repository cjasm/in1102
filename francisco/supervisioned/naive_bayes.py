# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from gaussian_multivariated import GaussianMultivariate

class GM:
    
    
    def preprocessing():
        #Importing the dataset
        dataset = pd.read_csv('../data/segmentation_data.csv')
        dataset_test = pd.read_csv('../data/segmentation_test.csv')
        X_shape = dataset.iloc[:, 1:10].drop(['REGION-PIXEL-COUNT'], axis=1).values
        X_rgb = dataset.iloc[:, 10:20].values
        y = dataset.iloc[:, 0].values
    
        # Encoding categorical data
        #Encoding the Dependent Variable
        labelencoder_y = LabelEncoder()
        y = labelencoder_y.fit_transform(y)
    
        # Feature Scaling
        sc_shape_X = StandardScaler()
        X_shape = sc_shape_X.fit_transform(X_shape)
        sc_rgb_X = StandardScaler()
        X_rgb = sc_rgb_X.fit_transform(X_rgb)
    
        # Importing the validating dataset
        X_shape_val = dataset_test.iloc[:, 1:10].drop(['REGION-PIXEL-COUNT'], axis=1).values
        X_rgb_val = dataset_test.iloc[:, 10:20]
        y_val = dataset_test.iloc[:, 0]
    
        # Encoding categorical data
        y_val = labelencoder_y.fit_transform(y_val)
    
        # Feature Scaling
        X_shape_val = sc_shape_X.fit_transform(X_shape_val)
        X_rgb_val = sc_rgb_X.fit_transform(X_rgb_val)
        
        return (X_shape, X_rgb, y),(X_shape_val, X_rgb_val, y_val)
    
    def train(X_shape, X_rgb, y):
        # Creating the K-fold stratified separation
        # 30 times(repeats) 10 fold(splits)
        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=0)

        gaussian_shape = GaussianMultivariate()
        gaussian_rgb = GaussianMultivariate()
        
        shape_proba = []
        for train_index, test_index in rskf.split(X_shape, y):
            gaussian_shape.fit(X_shape[train_index], y[train_index])
            y_pred = gaussian_shape.predict_proba(X_shape[test_index])
            shape_proba.append(y_pred)
        shape_proba = np.array(shape_proba)
        
        rgb_proba = []
        for train_index, test_index in rskf.split(X_rgb, y):
            gaussian_rgb.fit(X_rgb[train_index], y[train_index])
            y_pred = gaussian_rgb.predict_proba(X_rgb[test_index])
            rgb_proba.append(y_pred)
        rgb_proba = np.array(rgb_proba)
        
        sum_rule = np.zeros([300, rgb_proba.shape[1]])
        for j in range(300):
            all_scores_prob = np.zeros([rgb_proba.shape[1], rgb_proba.shape[2], 2])
            all_scores_prob[:, :, 0] = shape_proba[j]
            all_scores_prob[:, :, 1] = rgb_proba[j]
            y_pred_sum = []
            for i in range(rgb_proba.shape[1]):
              # Calculating the sum rule
              max_sum = np.amax(np.sum(all_scores_prob[i,:,:], axis=1))
              # Find the class that has the max sum
              y_pred_sum.append(np.where(np.sum(all_scores_prob[i,:,:], axis=1)==max_sum)[0][0])
            
            sum_rule[j,:] = np.array(y_pred_sum)
        
        i = 0
        accuracies = []
        for train_index, test_index in rskf.split(X_rgb, y):
            accuracy = accuracy_score(y[test_index], sum_rule[i,:])
            accuracies.append(accuracy)
            i += 1
            
        accuracy_means = []
        for i in range(30):
            accuracy_means.append(np.mean( accuracies[(0+i*10):(10+i*10)] ))
        
        return gaussian_shape, gaussian_rgb, accuracy_means
        
    def predict(gaussian_shape, gaussian_rgb, X_shape_val, X_rgb_val):
        # Sum Rule max1_r[(1-L)P(wr) + Pknnshape(wr|Xk) + Pknnviz(wr|Xk)]
        # The P(wr) is the same for any class, so it is irrelevant
        all_scores_prob = np.zeros([X_rgb_val.shape[0], 7, 2])
        all_scores_prob[:, :, 0] = gaussian_shape.predict_proba(X_shape_val)
        all_scores_prob[:, :, 1] = gaussian_rgb.predict_proba(X_rgb_val)
        y_pred_sum = []
        for i in range(X_rgb_val.shape[0]):
          # Calculating the sum rule
          max_sum = np.amax(np.sum(all_scores_prob[i,:,:], axis=1))
          # Find the class that has the max sum
          y_pred_sum.append(np.where(np.sum(all_scores_prob[i,:,:], axis=1)==max_sum)[0][0])
        
        return np.array(y_pred_sum)

    def predict_individual(gaussian_shape, gaussian_rgb, X_shape_val, X_rgb_val):
        y_shape_pred = gaussian_shape.predict(X_shape_val)
        y_rgb_pred = gaussian_rgb.predict(X_rgb_val)
        return y_shape_pred, y_rgb_pred
        
if __name__ == "__main__":
    training, validation = GM.preprocessing()
    gaussian_shape_clf, gaussian_rgb_clf, gaussian_accuracy_means = GM.train(training[0],training[1],training[2])
    gaussian_pred = GM.predict(gaussian_shape_clf, gaussian_rgb_clf, validation[0], validation[1])
    
    # Gaussian Individuals
    shape_pred, rgb_pred = GM.predict_individual(gaussian_shape_clf, gaussian_rgb_clf, validation[0], validation[1])
    print("-------- Gaussian Shape --------")
    print(confusion_matrix(validation[2], shape_pred))
    print(accuracy_score(validation[2], shape_pred))
    print(classification_report(validation[2], shape_pred))
    print("-------- Gaussian RGB --------")
    print(confusion_matrix(validation[2], rgb_pred))
    print(accuracy_score(validation[2], rgb_pred))
    print(classification_report(validation[2], rgb_pred))
    
    # Gaussian Metrics
    print("-------- Gaussian Sum --------")
    print(confusion_matrix(validation[2],gaussian_pred))
    print(classification_report(validation[2], gaussian_pred))
    gaussian_accuracy = accuracy_score(validation[2], gaussian_pred)
    print(gaussian_accuracy)
    # Wilson Score Confidence Interval
    # Constants values are 1.64 (90%) 1.96 (95%) 2.33 (98%) 2.58 (99%)
    const = 1.96
    n = training[1].shape[0]
    std = const * np.sqrt( (gaussian_accuracy * (1 - gaussian_accuracy)) / n)
    gaussian_ci = ((gaussian_accuracy - std), (gaussian_accuracy + std))
    print(gaussian_ci)