# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('segmentation_data.csv')
X_shape = dataset.iloc[:, 1:10].values
X_rgb = dataset.iloc[:, 10:20].values
y = dataset.iloc[:, 0].values

# Encoding categorical data
#Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_shape_X = StandardScaler()
X_shape = sc_shape_X.fit_transform(X_shape)
sc_rgb_X = StandardScaler()
X_rgb = sc_rgb_X.fit_transform(X_rgb)

# Creating the K-fold stratified separation
# 30 times(repeats) 10 fold(splits)
from sklearn.model_selection import RepeatedStratifiedKFold
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=0)

# Fitting Classifier to the Training Set
# Shape k = 3 and rgb = 1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
param = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10],
         'metric': ['minkowski'],
         'p':[2]}
grid_shape = GridSearchCV(KNeighborsClassifier(), param, cv=rskf)
grid_shape.fit(X_shape, y)
grid_rgb = GridSearchCV(KNeighborsClassifier(), param, cv=rskf)
grid_rgb.fit(X_rgb, y)

# Importing the validating dataset
dataset_test = pd.read_csv('segmentation_test.csv')
X_shape_val = dataset_test.iloc[:, 1:10]
X_rgb_val = dataset_test.iloc[:, 10:20]
y_val = dataset_test.iloc[:, 0]

# Encoding categorical data
y_val = labelencoder_y.fit_transform(y_val)

# Feature Scaling
X_shape_val = sc_shape_X.fit_transform(X_shape_val)
X_rgb_val = sc_rgb_X.fit_transform(X_rgb_val)

# Predicting the Test set Results using the validating dataset
y_shape_pred = grid_shape.predict(X_shape_val)
y_rgb_pred = grid_rgb.predict(X_rgb_val)

# Calculating the metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
shape_cm = confusion_matrix(y_val, y_shape_pred)
print(shape_cm)
print(accuracy_score(y_val, y_shape_pred))
print(classification_report(y_val, y_shape_pred))
rgb_cm = confusion_matrix(y_val, y_rgb_pred)
print(rgb_cm)
print(accuracy_score(y_val, y_rgb_pred))
print(classification_report(y_val, y_rgb_pred))

# Sum Rule max1_r[(1-L)P(wr) + Pknnshape(wr|Xk) + Pknnviz(wr|Xk)]
# The P(wr) is the same for any class, so it is irrelevant
all_scores_prob = np.zeros([X_rgb_val.shape[0], 7, 2])
all_scores_prob[:, :, 0] = grid_shape.predict_proba(X_shape_val)
all_scores_prob[:, :, 1] = grid_rgb.predict_proba(X_rgb_val)
y_pred_sum = []
for i in range(X_rgb_val.shape[0]):
    # Calculating the sum rule
    max_sum = np.amax(np.sum(all_scores_prob[i,:,:], axis=1))
    # Find the class that has the max sum
    y_pred_sum.append(np.where(np.sum(all_scores_prob[i,:,:], axis=1)==max_sum)[0][0])
    
y_pred_sum=np.array(y_pred_sum)

# Calculating the classifiers combination dataset
sum_cm = confusion_matrix(y_val,y_pred_sum)
print(accuracy_score(y_val, y_pred_sum))
print(classification_report(y_val, y_pred_sum))

accuracy = accuracy_score(y_val, y_pred_sum)
error = 1 - accuracy
# Wilson Score Confidence Interval
# Constants values are 1.64 (90%) 1.96 (95%) 2.33 (98%) 2.58 (99%)
const = 1.96
n = X_rgb_val.shape[0]
std = const * np.sqrt( (error * (1 - error)) / n)
ci = (error - std, error + std)

