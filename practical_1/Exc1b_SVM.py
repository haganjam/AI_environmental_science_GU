# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 14:13:53 2021

@author: Heather
"""

import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import pandas as pd
import numpy as np
#import os as os

# to make the output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from sklearn.svm import SVC
from sklearn import datasets #Note that sklearn has a few built-in datasets, including Iris. See https://scikit-learn.org/stable/datasets/toy_dataset.html
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#Imported from sklearn, the data is always a 2D array, shape (n_samples, n_features) as a .data member
#Note that we have not read the data into a dataframe
iris = datasets.load_iris()

#We will load only petal characteristics
#Remember that SVM is in its basic form, only a binary classifier
#Therefore we will start with just simple classification
X = iris["data"][:, (2, 3)]  # where column 0 is sepal length, 1 is sepal width, 2 is petal length, 3 is petal width, all in cm
y = (iris["target"] == 2).astype(np.float64)  #where 0 is Setosa, 1 is Versicolor and 2 is Virginica

#If you just want to check the data
print(iris.target)

#The following Scikit-Learn code scales the features, and then trains a linear SVM model (using the LinearSVC  class with C=1  and the hinge loss function, described shortly) to detect Iris virginica  flowers
svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ])

#
svm_clf.fit(X, y)

#Apply your SVM model to predict Iris species from the following petal length and width
test1 = svm_clf.predict([[5.5, 1.7]])
print(test1)


####PLOT ####################
###To see how this model fits with your data at C=1 and C=100, plot the following
scaler = StandardScaler()
# Here you can set two different Cs
svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

scaled_svm_clf1 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf1),
    ])
scaled_svm_clf2 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ])

scaled_svm_clf1.fit(X, y)
scaled_svm_clf2.fit(X, y)

# Convert to unscaled parameters
b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
w1 = svm_clf1.coef_[0] / scaler.scale_
w2 = svm_clf2.coef_[0] / scaler.scale_
svm_clf1.intercept_ = np.array([b1])
svm_clf2.intercept_ = np.array([b2])
svm_clf1.coef_ = np.array([w1])
svm_clf2.coef_ = np.array([w2])

# Find support vectors (LinearSVC does not do this automatically)
t = y * 2 - 1
support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
svm_clf1.support_vectors_ = X[support_vectors_idx1]
svm_clf2.support_vectors_ = X[support_vectors_idx2]

fig, axes = plt.subplots(ncols=2, figsize=(10,2.7), sharey=True)

###CREATE FUNCTION TO PLOT DECISION BOUNDARY
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
### END FUNCTION

plt.sca(axes[0])
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris virginica")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Iris versicolor")
plot_svc_decision_boundary(svm_clf1, 4, 5.9)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)
plt.axis([4, 5.9, 0.8, 2.8])

plt.sca(axes[1])
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
plot_svc_decision_boundary(svm_clf2, 4, 5.99)
plt.xlabel("Petal length", fontsize=14)
plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
plt.axis([4, 5.9, 0.8, 2.8])
##### END PLOT ################################

### NON-LINEAR SVM
# The following function creates a dataset that is more complex and non-linear
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()

# You can apply a polynomial kernel which gives you the same result ...
# ...as if you had added several polynomial features
# There are several hyperparameters that are of importance here 
# namely C, the degree of the polynomal, and coef0
# The hyperparameter coef0  controls how much the model is influenced by highdegree polynomials versus low-degree polynomials.

### Degree = 3, Coef0 = 1 and C=5
poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X, y)

####
##### TEST WHAT HAPPENS WHEN YOU CHANGE THE HYPERPARAMETERS
# Let's run the same thing, but with a 10th degree polynomial and coef=100

### Degree = 10, Coef0 = 100 and C=5
poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])
poly100_kernel_svm_clf.fit(X, y)


#### K-FOLD CROSS-VALIDATION TO ASSESS RESULTS 
# Perform a 5-fold cross-validation on the poly_kernel dataset
from sklearn.model_selection import cross_val_score
scores = cross_val_score(poly_kernel_svm_clf, X, y,
                         scoring="neg_mean_squared_error", cv=5)
poly_k_rmse_scores = np.sqrt(-scores)

print("RMSE for d=3, Coef0=1 and C=5")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())

scores100 = cross_val_score(poly100_kernel_svm_clf, X, y,
                         scoring="neg_mean_squared_error", cv=5)
poly100_k_rmse_scores = np.sqrt(-scores)

print("RMSE for d=10, Coef0=100 and C=5")
print("Scores:", scores100)
print("Mean:", scores100.mean())
print("Standard deviation:", scores100.std())
######


#### PLOT
# First a function for plotting predictions
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
# End function

fig, axes = plt.subplots(ncols=2, figsize=(10.5, 4), sharey=True)

plt.sca(axes[0])
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.45, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)

plt.sca(axes[1])
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.45, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)
plt.ylabel("")

plt.show()


#### GRID SEARCH FOR TESTING DIFFERENT HYPERPARAMETERS
# Of course this is not an efficient way to determine hyperparameter values
# Using a Grid search is more efficient
# The inputs are which hyperparameters you want to experiment with
# And what values of the hyperparameters you want to try

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 27 (3×3x3) combinations of hyperparameters
    {'degree': [3, 5, 10], 'C': [5, 10, 100], 'coef0':[1, 5, 50]}
      ]

#define a variable for the generic function without hyperparameters
svm1 = SVC(kernel="poly")
# Train across 5-folds, which is a total of (27+8)*5= 165 rounds of training
grid_search_svm = GridSearchCV(svm1, param_grid, cv=5)
grid_search_svm.fit(X,y)


#Output the best hyperparameter combination found
grid_search_svm.best_params_
#grid_search_svm.best_estimator_

pd.DataFrame(grid_search_svm.cv_results_)

###### END OF GRID SEARCH #######

##### GAUSSIAN RBF Kernel
# Another useful approach in non-linear SVM is to use a kernel that employs a "similarity feature"
#This employs a Gaussian Radial Basis Function Kernel (or RBF)
# The hyperparameters to adjust are gamma and C
rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])
rbf_kernel_svm_clf.fit(X, y)

#Let's try some different parameters and plot them
gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 7), sharex=True, sharey=True)

for i, svm_clf in enumerate(svm_clfs):
    plt.sca(axes[i // 2, i % 2])
    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.45, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)
    if i in (0, 1):
        plt.xlabel("")
    if i in (1, 3):
        plt.ylabel("")

plt.show()


#### Short EXAMPLE OF SVM Regression
# Linear SVM regression
from sklearn.svm import LinearSVR

m = 50
X = 2 * np.random.rand(m, 1)
y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

svm_reg = LinearSVR(epsilon=1.5, random_state=42)
svm_reg.fit(X, y)

# Non-linear SVM regression
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
svm_poly_reg.fit(X, y)
