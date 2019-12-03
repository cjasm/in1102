# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from naive_bayes import GM
from knn import KNN

# Gaussian Naive Bayes
training, validation = GM.preprocessing()
gaussian_shape_clf, gaussian_rgb_clf, gaussian_accuracy_means = GM.train(training[0], training[1], training[2])
gaussian_pred = GM.predict(gaussian_shape_clf, gaussian_rgb_clf, validation[0], validation[1], )

# Gaussian Metrics
print("-------- Gaussian --------")
print("Confusion Matrix")
print(confusion_matrix(validation[2],gaussian_pred))
print("Classification Report")
print(classification_report(validation[2], gaussian_pred))
print("Validation Accuracy:", accuracy_score(validation[2], gaussian_pred))

# Wilson Score Confidence Interval
# Constants values are 1.64 (90%) 1.96 (95%) 2.33 (98%) 2.58 (99%)
gaussian_accuracy = np.random.choice(gaussian_accuracy_means)
const = 1.96
n = training[1].shape[0]
std = const * np.sqrt( (gaussian_accuracy * (1 - gaussian_accuracy)) / n)
gaussian_ci = ((gaussian_accuracy - std), (gaussian_accuracy + std))
print("Ponctual Estimation", gaussian_accuracy)
print("Confidence Interval:", gaussian_ci)

# KNN
training, validation = KNN.preprocessing()
knn_shape_clf, knn_rgb_clf, knn_accuracy_means = KNN.train(training[0], training[1], training[2])
knn_pred = KNN.predict(knn_shape_clf, knn_rgb_clf, validation[0], validation[1])

# KNN Metrics
print("-------- KNN --------")
print("Confusion Matrix")
print(confusion_matrix(validation[2],knn_pred))
print("Classification Report")
print(classification_report(validation[2], knn_pred))
print("Validation Accuracy:", accuracy_score(validation[2], knn_pred))
# Wilson Score Confidence Interval
# Constants values are 1.64 (90%) 1.96 (95%) 2.33 (98%) 2.58 (99%)
knn_accuracy = np.random.choice(knn_accuracy_means)
const = 1.96
n = training[1].shape[0]
std = const * np.sqrt( (knn_accuracy * (1 - knn_accuracy)) / n)
knn_ci = ((knn_accuracy - std), (knn_accuracy + std))
print("Ponctual Estimation", knn_accuracy)
print("Confidence Interval:", knn_ci)

# Overall Metrics
stat, p = wilcoxon(gaussian_accuracy_means, knn_accuracy_means)
print("-------- Gaussian x KNN --------")
print('Wilcoxon Statistics=%.3f, p=%.7f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')