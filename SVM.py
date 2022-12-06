import numpy as np
import time
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


# The example from HW5 Problem 2


X = np.array([[0, 0], [0, -1], [-2, 0]]) # 3 x 2
y = np.array([-1, -1, 1]).reshape(3, 1)  # 3 x 1

dim = X.shape[1]  # dimensionality
num = X.shape[0]  # sample size

Q = np.eye(dim+1)
Q[0, 0] = 0
p = np.zeros((dim+1, 1))
A = np.array(np.concatenate((y, y * X), axis=1)).astype(float)
c = np.ones((num, 1))

# Code below removes the progress printing out
solvers.options["show_progress"] = False

sol = solvers.qp(P=matrix(Q), q=matrix(p), G=matrix(-A), h=matrix(-c))
print("Solution for Problem 2 (Test Run):")
print("The first value is b* and the rest of the values correspond to w*")
print(sol['x'])


# Question 3 Part A: Using the toy data from the class lecture slides (Lecture 11 Slide 21):


X = np.array([[0, 0], [2, 2], [2, 0], [3, 0]])  # 4 x 2
y = np.array([-1, -1, 1, 1]).reshape(4, 1)  # 4 x 1

dim = X.shape[1]  # dimensionality
num = X.shape[0]  # sample size

Q = np.eye(dim+1)
Q[0, 0] = 0
p = np.zeros((dim+1, 1))
A = np.array(np.concatenate((y, y * X), axis=1)).astype(float)
c = np.ones((num, 1))

sol = solvers.qp(P=matrix(Q), q=matrix(p), G=matrix(-A), h=matrix(-c))
print("Solution for the toy data from the class lecture slides (Lecture 11 Slide 21):")
print("The first value is b* and the rest of the values correspond to w*")
print(sol['x'])


# Question 3 Part B: 

def generatePoints(feature_size, sample_size):
    if(sample_size < feature_size):
        raise ValueError("Error: Sample size must be greater than or equal to the feature size!")
    X = np.random.randn(sample_size, feature_size)  # Randomly Generated Data
    w = np.random.randn(feature_size, 1)    # Fake "Ground Truth" Model to Assign Labels Based on Linear Separation
    y = np.ones(sample_size)
    avg = 0

    for i in range(sample_size):
        avg += np.dot(X[i], w)
    
    avg = avg / sample_size

    for i in range(sample_size):
        a = np.dot(X[i], w)
        if a < avg:
            y[i] = -1
        else:
            y[i] = 1
    
    y = y.reshape(sample_size, 1)
    return X, y


# Ensure that sample_size > feature_size
X, y = generatePoints(10, 100)

dim = X.shape[1]  # dimensionality
num = X.shape[0]  # sample size

Q = np.eye(dim+1)
Q[0, 0] = 0
p = np.zeros((dim+1, 1))
A = np.array(np.concatenate((y, y * X), axis=1)).astype(float)
c = np.ones((num, 1))

sol = solvers.qp(P=matrix(Q), q=matrix(p), G=matrix(-A), h=matrix(-c))

print("Dry run (test run) solution for the randomly generated, linearly separable data:")
print("The first value is b* and the rest of the values correspond to w*")
print(sol['x'])
print()


# How does time cost grow as sample_size increases (with a fixed feature_size)?


features = 1000
sampleSizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
times = []

print("Testing execution time as feature size is fixed and sample size varies:")
for i in range(len(sampleSizes)):
    X, y = generatePoints(features, sampleSizes[i])
    t0 = time.time()

    dim = X.shape[1]  # dimensionality
    num = X.shape[0]  # sample size

    Q = np.eye(dim+1)
    Q[0, 0] = 0
    p = np.zeros((dim+1, 1))
    A = np.array(np.concatenate((y, y * X), axis=1)).astype(float)
    c = np.ones((num, 1))

    print("Testing sample size = " + str(sampleSizes[i]) + ":")
    sol = solvers.qp(P=matrix(Q), q=matrix(p), G=matrix(-A), h=matrix(-c))

    # Uncomment the following 2 lines if you want to view the results of the solutions. I commented these out because it was flooding the terminal
    # print("The first value is b* and the rest of the values correspond to w*")
    # print(sol['x'])

    t1 = time.time()
    times.append(t1 - t0)
    print("Time Taken For This Iteration = " + str(t1 - t0) + " seconds")

plt.plot(sampleSizes, times)
plt.title("Time to solve the hard margin SVM problem with a feature size of " + str(features) + " and varying sample sizes")
plt.xlabel("Sample Size")
plt.ylabel("Time Taken In Seconds To Solve The Problem")
plt.show()
print()


# How does time cost grow as feature_size increases (with a fixed sample_size)?


samples = 5000
featureSizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
times = []

print("Testing execution time as sample size is fixed and feature size varies:")
for i in range(len(featureSizes)):
    X, y = generatePoints(featureSizes[i], samples)
    t0 = time.time()

    dim = X.shape[1]  # dimensionality
    num = X.shape[0]  # sample size

    Q = np.eye(dim+1)
    Q[0, 0] = 0
    p = np.zeros((dim+1, 1))
    A = np.array(np.concatenate((y, y * X), axis=1)).astype(float)
    c = np.ones((num, 1))

    print("Testing feature size = " + str(featureSizes[i]) + ":")
    sol = solvers.qp(P=matrix(Q), q=matrix(p), G=matrix(-A), h=matrix(-c))

    # Uncomment the following 2 lines if you want to view the results of the solutions. I commented these out because it was flooding the terminal
    # print("The first value is b* and the rest of the values correspond to w*")
    # print(sol['x'])

    t1 = time.time()
    times.append(t1 - t0)
    print("Time Taken For This Iteration = " + str(t1 - t0) + " seconds")

plt.plot(featureSizes, times)
plt.title("Time to solve the hard margin SVM problem with a sample size of " + str(samples) + " and varying feature sizes")
plt.xlabel("Feature Size")
plt.ylabel("Time Taken In Seconds To Solve The Problem")
plt.show()