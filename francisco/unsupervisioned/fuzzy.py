# -*- coding: utf-8 -*-
# C-means Fuzzy

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel


# Importing the dataset
def importing_data():
    dataset = pd.read_csv('../data/segmentation_data.csv')
    shape = dataset.iloc[:, 1:10].values
    rgb = dataset.iloc[:, 10:20].values
    return shape, rgb

# Fuzzy Model
def fuzzy_model(X):
    c=7 # number of clusters
    n=X.shape[0] # number of data
    p=X.shape[1] # number of dimensions
    m=1.6
    T=150
    e=10**(-10)
    t=1
    objective_value_old=0
    objective_value_new=0

    while( not(objective_value_new - objective_value_old <= e) or not(t>T)):
        updating_centroid()
        updating_weights()
        updating_fuzzy_membership_degree()
        objective_value_old = objective_value_new
        objective_value_new = objective()
        t += 1

# Random initializing the fuzzy membership degrees
def initializing_membership_degree(c, n):
    """
    c: number of clusters
    n: number of samples
    """
    fuzzy_membership_degree = np.random.rand(c,n)
    fuzzy_membership_degree = fuzzy_membership_degree/fuzzy_membership_degree.sum(axis=0, keepdims=1)
    return fuzzy_membership_degree

# Initializing the weights vector
def initializing_weights(p):
    """
    p: number of dimensions
    """
    weights = np.ones(p)
    return weights

# Random initializing the center of the clusters
def initializing_centroids(X, c):
    """
    X: data sample
    c: number of clusters
    """
    centroids = np.array([random.choice(X) for x in range(c)])
    return centroids

# Calculating variance through quantile
def two_sigma_square_for_dimension(X, j):
    """
    X: data sample
    j: dimension position
    """
    result = []
    for k in range(X.shape[0]):
        for l in range(X.shape[1]):
            if not(l==k):
                difference = euclidean_distances(X[k,j], X[k,l])**2
                result.append(difference)
    result = np.array(sorted(result))
    mean = np.mean([np.quantile(result,0.1),np.quantile(result,0.9)])
    return mean


# v_ij = sum(u_ik*K(x_kj, v_ij)*x_kj)/sum(u_ik*K(x_kj, v_ij))
def updating_centroid(i, j, u, X, m, v):
    """
    i: cluster position
    k: sample position
    u: fuzzy membership degree
    X: data sample
    m: constant
    v: centroid
    """
    nominators = []
    denominators = []
    quantile = 1/two_sigma_square_for_dimension(X, j)
    for k in range(X.shape[0]):
        kernel = rbf_kernel(X[k, j], v[i, j], gamma=quantile)
        nominators.append(
                (u[i,k]**m) * kernel * X[k,j])
        denominators.append(
                (u[i,k]**m) * kernel)

    return nominators.sum()/denominators.sum()

def updating_weights(weights, X, u, v, j, m):
    """
    weights: weights vector
    X: data sample
    u: fuzzy membership degree
    v: centroid
    j: weight vector dimension to update
    """
    p=weights.shape[0]
    nominators = []
    for l in range(p):
        c_sum = 0
        quantile = 1/two_sigma_square_for_dimension(X, l)
        for i in range(v.shape[0]):
            k_sum = 0
            for k in range(X.shape[0]):
                kernel = rbf_kernel(X[k, l], v[i, l], gamma=quantile)
                k_sum += (u[i,k]**m) * (2*(1-kernel))
            c_sum += k_sum
        nominators.append(c_sum)

    denominators = []
    quantile = 1/two_sigma_square_for_dimension(X, j)
    for i in range(v.shape[0]):
            k_sum = 0
            for k in range(X.shape[0]):
                kernel = rbf_kernel(X[k, j], v[i, j], gamma=quantile)
                k_sum += (u[i,k]**m) * (2*(1-kernel))
            denominators.append(k_sum)
    nominator = np.array(nominators).prod()**1/p
    denominator = np.array(denominators).sum()
    return nominator / denominator

def updating_fuzzy_membership_degree():
    pass

def global_adaptative_distance(weights, X, v, k, i):
    """
    weights: weights vector
    X: sample
    v: centroid
    k: sample position
    i: cluster position
    """
    p = weights.shape[0]
    distance = 0
    for j in range(p):
        quantile = 1/two_sigma_square_for_dimension(X, j)
        kernel = rbf_kernel(X[k, j], v[i, j], gamma=quantile)
        distance = weights[j] * (2*(1-kernel))
    return distance

def objective():
    pass







if __name__== "__main__" :
    for X in importing_data():
        fuzzy_model(X)