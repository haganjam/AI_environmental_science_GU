# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 18:44:33 2021

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

# to make the output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline #Only for Jupyter Notebook
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from sklearn.svm import SVC
from sklearn import datasets 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()

X = iris["data"] #where 0 is sepal length, 1 is sepal width, 2 is petal length, 3 is petal width, all in cm
y = (iris["target"].astype(np.float64))  #where 0 is Setosa, 1 is Versicolor and 2 is Virginica

#We can create a test and training dataset using scikitlearns function
# We will split it 80% to train, 20% to test, and shuffle the data since training can be sensitive to the order of the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Here we will train the model. 
# If there are more than one class, Scikitlearn automatically recognizes this
# It switches to a One-vs-the-Rest strategy. You don't have to change anything. 

lin_clf=LinearSVC(random_state=42)
lin_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = lin_clf.predict(X_train)
accuracy_score(y_train, y_pred)

# REMEMBER THAT IN SOME CASES THE DATA NEED TO BE SCALED. YOU MIGHT TRY THAT GIVEN OTHER DATA SETS. 
# THE Iris data do not require scaling...

svm_clf = SVC(gamma="scale")
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_train)
accuracy_score(y_train, y_pred)

#Do a Grid search to find the best parameters
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2, cv=3)
rnd_search_cv.fit(X_train, y_train)

rnd_search_cv.best_score_
rnd_search_cv.best_estimator_

# Lets apply the model with the best estimators suggested
rnd_search_cv.best_estimator_.fit(X_train, y_train)
y_pred = rnd_search_cv.best_estimator_.predict(X_train) 
accuracy_score(y_train, y_pred)

# The cross-validation says its a decent model, so let's apply it to the test set
y_pred = rnd_search_cv.best_estimator_.predict(X_test)
accuracy_score(y_test, y_pred)

plt.scatter(y_test, y_pred)
plt.show()

#This was a little bit boring with the Iris data set
# How about trying it with another dataset? 
