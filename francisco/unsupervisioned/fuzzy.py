# -*- coding: utf-8 -*-
# C-means Fuzzy

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import adjusted_rand_score
from datetime import datetime


# Importing the dataset
def importing_data():
    dataset = pd.read_csv('../data/segmentation_test.csv')
    shape = dataset.iloc[:, 1:10].drop(['REGION-PIXEL-COUNT'], axis=1).values
    rgb = dataset.iloc[:, 10:20].values
    y = dataset.iloc[:, 0].values
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    return shape, rgb, y
    
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
    centroids = np.array([random.choice(X) for _ in range(c)])
    return centroids

# Calculating variance through quantile
def two_sigma_square(X, p):
    variances = []
    for j in range(p):
        variances.append(two_sigma_square_for_dimension(X, j))
    
    return np.array(variances)

def two_sigma_square_for_dimension(X, j):
    """
    X: data sample
    j: dimension position
    """
    result = []
    for k in range(X.shape[0]):
        for l in range(X.shape[1]):
            if not(l==k):
                difference = np.power(abs(X[k,j] - X[k,l]),2)
                result.append(difference)
    result = np.array(sorted(result))
    mean = np.mean([np.quantile(result,0.1),np.quantile(result,0.9)])
    return float(mean)

def updating_centroids(u, X, m, v, variances):
    """
    u: fuzzy membership degree
    X: data sample
    m: constant
    v: centroid
    """
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            v[i,j] = updating_centroid(i, j, u, X, m, v, variances)
    return v

# v_ij = sum(u_ik*K(x_kj, v_ij)*x_kj)/sum(u_ik*K(x_kj, v_ij))
def updating_centroid(i, j, u, X, m, v, variances):
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
    quantile = np.divide(1, variances[j])
    for k in range(X.shape[0]):
        kernel = rbf_kernel(X[k, j].reshape(-1, 1),
                            v[i, j].reshape(-1, 1), gamma=quantile)
        nominators.append(
                np.multiply(np.multiply(np.power(u[i,k], m), kernel), X[k,j]))
        denominators.append(
                np.multiply(np.power(u[i,k], m), kernel))
    nominators = np.array(nominators)
    denominators = np.array(denominators)
    return nominators.sum()/denominators.sum()

def updating_weights(weights, X, u, v, m, variances):
    """
    weights: weights vector
    X: data sample
    u: fuzzy membership degree
    v: centroid
    m: constant
    """
    for j in range(weights.shape[0]):
        weights[j] = updating_weight(j, weights, X, u, v, m, variances)
    return weights

def updating_weight(j, weights, X, u, v, m, variances):
    """
    j: weight vector dimension to update
    weights: weights vector
    X: data sample
    u: fuzzy membership degree
    v: centroid
    m: constant
    """
    p=weights.shape[0]
    nominators = []
    for l in range(p):
        c_sum = 0
        quantile = np.divide(1, variances[l])
        for i in range(v.shape[0]):
            k_sum = 0
            for k in range(X.shape[0]):
                kernel = rbf_kernel(X[k, l].reshape(-1, 1),
                                    v[i, l].reshape(-1, 1), gamma=quantile)
                k_sum += np.multiply(np.power(u[i,k], m),
                                     np.multiply(2, 1-kernel))
            c_sum += k_sum
        nominators.append(c_sum)

    denominator = 0.0
    quantile = np.divide(1, variances[j])
    for i in range(v.shape[0]):
            k_sum = 0
            for k in range(X.shape[0]):
                kernel = rbf_kernel(X[k, j].reshape(-1, 1),
                                    v[i, j].reshape(-1, 1), gamma=quantile)
                k_sum += np.multiply(np.power(u[i,k], m),
                                     np.multiply(2, 1-kernel))
            denominator += k_sum
            
    nominator = np.power(np.array(nominators).prod(), 1/float(p))
    return nominator / denominator

def global_adaptative_distance(weights, X, v, k, i, variances):
    """
    weights: weights vector
    X: sample
    v: centroid
    k: sample position
    i: cluster position
    variances: variances
    """
    p = weights.shape[0]
    distance = 0
    for j in range(p):
        quantile = np.divide(1, variances[j])
        kernel = rbf_kernel(X[k, j].reshape(-1, 1),
                            v[i, j].reshape(-1, 1), gamma=quantile)
        distance = np.multiply(weights[j], np.multiply(2, 1-kernel))
    return distance

def updating_fuzzy_memberships_degree(u, X, v, weights, m, variances):
    """
    u: membership degree
    X: data sample
    v: centroids
    m: constant
    weights: weights vector
    """
    for i in range(u.shape[0]):
        for k in range(u.shape[1]):
            u[i,k] = updating_fuzzy_membership_degree(i, k, X, v, weights, m, variances)
    return u

def updating_fuzzy_membership_degree(i, k, X, v, weights, m, variances):
    """
    i: cluster position
    k: sample position
    X: data sample
    v: centroids
    m: constant
    weights: weights vector
    """
    result = 0.0
    numerator = global_adaptative_distance(weights, X, v, k, i, variances)
    for h in range(v.shape[0]):
     denominator = global_adaptative_distance(weights, X, v, k, h, variances)
     result += np.power(np.divide(numerator,denominator), np.divide(1,(m-1)))
    result = np.divide(1.0, result)

    return result


def objective(X, u, v, weights, m, variances):
    """
    X: data sample
    u: fuzzy membership degree
    v: centroids
    weights: weights vector
    m: constant
    """
    value = 0
    for i in range(v.shape[0]):
        parcial = 0
        for k in range(v.shape[1]):
            parcial += np.multiply(np.power(u[i,k], m), 
                                   global_adaptative_distance(weights, X, v, k, i, variances))
        value += parcial

    return value

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
    membership = initializing_membership_degree(c, n)
    centroids = initializing_centroids(X, c)
    weights = initializing_weights(p)
    variances = two_sigma_square(X, p)

    while( abs(objective_value_new - objective_value_old) > e or t<=T):
        centroids = updating_centroids(membership, X, m, centroids, variances)
        # print("Centroids:", centroids)
        weights = updating_weights(weights, X, membership, centroids, m, variances)
        # print("Weights:", weights)
        # print("Weights Prod:", weights.prod())
        membership = updating_fuzzy_memberships_degree(membership, X, centroids, weights, m, variances)
        # membership_sum = []
        # for i in range(n):
        #     membership_sum.append(membership[:,i].sum())
        # print("Membership Sum:", membership_sum)
        objective_value_old = objective_value_new
        objective_value_new = objective(X, membership, centroids, weights, m, variances)
        # print(t)
        t += 1
        # print("Objective:", objective_value_new)
     
    return membership, weights, centroids, objective_value_new

if __name__== "__main__" :
    X_shape, X_rgb, y = importing_data()
    
    # Shape view
    shape_objective = 1
    shape_membership = np.array([])
    shape_weights = np.array([])
    shape_centroids = np.array([])
    # RGB view
    rgb_objective = 1
    rgb_membership = np.array([])
    rgb_weights = np.array([])
    rgb_centroids = np.array([])
    
    for i in range(100):
        start = datetime.now()
        # Shape view
        membership, weights, centroids, objective_value = fuzzy_model(X_shape)
        if objective_value < shape_objective:
            shape_objective = objective_value
            shape_membership = membership
            shape_weights = weights
            shape_centroids = centroids
            
            # RGB view
        membership, weights, centroids, objective_value = fuzzy_model(X_rgb)
        if objective_value < rgb_objective:
            rgb_objective = objective_value
            rgb_membership = membership
            rgb_weights = weights
            rgb_centroids = centroids
        time_taken = datetime.now() - start
        print("Round:", i, "-", "Time Spent:", time_taken)
    
    print("------------ SHAPE ------------")
    print("SHAPE minimum objective value:", shape_objective)
    print("SHAPE centroids:", shape_centroids)
    print("SHAPE weights:", shape_weights)
    shape_crisp = shape_membership.argmax(axis=0)
    print("SHAPE Rand Score:", adjusted_rand_score(shape_crisp, y))
    for i in range(7):
        print("Cluster", i)
        elements = np.where(shape_crisp==i)[0]
        print("Count Elements:", len(elements))
        print("Elements:", elements)
        print("")

    print("------------ RGB ------------")
    print("RGB minimum objective value:", rgb_objective)
    print("RGB centroids:", rgb_centroids)
    print("RGB weights:", rgb_weights)
    rgb_crisp = rgb_membership.argmax(axis=0)
    print("RGB Rand Score:", adjusted_rand_score(rgb_crisp, y))
    for i in range(7):
        print("Cluster", i)
        elements = np.where(rgb_crisp==i)[0]
        print("Count Elements:", len(elements))
        print("Elements:", elements)
        print("")
    
    print("------------ SHAPE x RGB ------------")
    print("SHAPE X RGB - Rand Score:", adjusted_rand_score(shape_crisp, rgb_crisp))
    
    print("FINISHED")
