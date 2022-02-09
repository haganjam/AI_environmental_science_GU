
### AI Environmental Science: Practical 2 Neural networks

# import necessary libraries
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
tbl = pd.read_csv("C:/Users/james/OneDrive/PhD_Gothenburg/PhD_courses/AI_earth_environmental_science/Practicals/Practical_2_neural_networks/div_data_all_features.txt", delimiter="\t")
tbl.head()


## define the model features

# extract an array of feature values
features = tbl.values[:, 1:]

# extract the feature names
feature_names = tbl.columns[1:]
print(feature_names)

# get the feature labels
labels = tbl.values[:,0]

# plot the histogram of labels
plt.hist(labels)
plt.show()


## rescale the variables

# define a function to rescale the data array between 0 and 1
def min_max_scaler(x, min_array=None, max_array=None, inverse_scale=False):
    if inverse_scale:
        x_new = x*(max_array-min_array)+min_array
        return(x_new)
    else:
        min_array = np.min(x,axis=0)
        max_array = np.max(x,axis=0)
        x_new = (x - min_array)/(max_array - min_array)
        return(x_new,min_array,max_array)

# rescale the features in the feature array
rescaled_features, scale_min, scale_max = min_max_scaler(features)

# this function outputs the rescaled features, the min and max used for each feature
print(scale_min, scale_max)

# rescale the labels
rescaled_labels = labels/800
plt.hist(rescaled_labels)
plt.show()


## separate the instances into the different data sets

# define a function to selecct training, validation and test data
def select_train_val_test(x,val_fraction=0.2,test_fraction=0.2,
                          shuffle=True,seed=None):
    
    all_indices = np.arange(len(x))
    if shuffle:
        if not seed:
            seed = np.random.randint(0,999999999)
            
        # shuffle all input data and labels
        np.random.seed(seed)
        print('Shuffling data, using seed', seed)
        shuffled_indices = np.random.choice(all_indices, len(all_indices), replace=False)
   
    else:
        shuffled_indices = all_indices
        
    # select train, validation, and test data
    n_test_instances = np.round(len(shuffled_indices) * test_fraction).astype(int)
    n_validation_instances = np.round(len(shuffled_indices) * val_fraction).astype(int)
    test_ids = shuffled_indices[:n_test_instances]
    validation_ids = shuffled_indices[n_test_instances:n_test_instances + n_validation_instances]
    train_ids = shuffled_indices[n_test_instances + n_validation_instances:]
    return train_ids, validation_ids, test_ids


# separate instances into train and test set
# note that we need separate datasets for the features and the labels
train_set_ids, validation_set_ids, test_set_ids = select_train_val_test(rescaled_features)
train_features = rescaled_features[train_set_ids]
train_labels = rescaled_labels[train_set_ids]
validation_features = rescaled_features[validation_set_ids]
validation_labels = rescaled_labels[validation_set_ids]
test_features = rescaled_features[test_set_ids]
test_labels = rescaled_labels[test_set_ids]


## define the neural network model

architecture = []

# input layer
architecture.append(tf.keras.layers.Flatten(input_shape = [train_features.shape[1]]) )

# 1st hidden layer: 32 nodes and relu activation function
architecture.append(tf.keras.layers.Dense(32, activation = "relu"))

# 2nd hidden layer: 8 nodes and relu activation function
architecture.append(tf.keras.layers.Dense(8, activation = "relu"))

# ouput layer: 1 output node (regression) and softplus activation function
architecture.append(tf.keras.layers.Dense(32, activation = "softplus"))

# compile the model
model = tf.keras.Sequential(architecture)
model.compile(loss = "mae", optimizer = "adam", metrics = ["mae"])

# get overview of the model's architecture
model.summary()














