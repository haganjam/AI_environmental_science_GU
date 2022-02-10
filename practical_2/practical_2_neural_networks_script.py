
### AI Environmental Science: Practical 2 Neural networks

# import necessary libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
tbl = pd.read_csv("C:/Users/james/OneDrive/PhD_Gothenburg/PhD_courses/AI_earth_environmental_science/Practicals/Practical_2_neural_networks/div_data_all_features.txt", delimiter="\t")
tbl.head()
tbl.shape


### Neural networks

## define the model features

# extract an array of feature values
features = tbl.values[:, 1:]
features.shape

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
architecture.append(tf.keras.layers.Dense(1, activation = "softplus"))

# compile the model
model = tf.keras.Sequential(architecture)
model.compile(loss = "mae", optimizer = "adam", metrics = ["mae"])

# get overview of the model's architecture
model.summary()

# make predictions without training the model
estimated_test_labels = model.predict(test_features)

# plot these predictions
def plot_true_vs_pred(true_labels, predicted_labels):
    fig = plt.figure(figsize=(6, 6))
    plt.plot(true_labels, predicted_labels, 'o', markersize=3, alpha=1)
    plt.plot([0,800],[0,800],'r-')
    plt.grid()
    plt.xlabel('True diversity')
    plt.ylabel('Predicted diversity')

# this doesn't look right but let's see
plot_true_vs_pred(test_labels*800,estimated_test_labels*800)


## training the model

# define early stop of training based on validation set
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae',patience=200,restore_best_weights=True)

# run model training and store training history
history = model.fit(train_features,
                    train_labels,
                    epochs=3000,
                    validation_data=(validation_features, validation_labels), 
                    verbose=1,
                    callbacks=[early_stop],
                    batch_size=40) # this is how pieces the data are split into

# define a function to plot the model
def plot_training_history(history, show_best_epoch=True):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history.history['mae'], label='Training set')
    plt.plot(history.history['val_mae'], label='Validation set') 
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    if show_best_epoch:
        best_epoch = np.where(history.history['val_mae'] == np.min(history.history['val_mae']))[0][0]
        plt.axvline(best_epoch, c='grey', linestyle='--')
        plt.axhline(history.history['val_mae'][best_epoch], c='grey', linestyle = '--')
        plt.gca().axvspan(best_epoch, len(history.history['mae']),
                          color = 'grey', alpha = 0.3, zorder = 3)
        plt.grid()
        plt.legend(loc = "upper center")

# plot
plot_training_history(history)   

# save the model so you don't have to run it again
model_file = "/OneDrive/PhD_Gothenburg/PhD_courses/AI_earth_environmental_science/Practicals/Practical_2_neural_networks/trained_model"
model.save(model_file)
# This is how you load the model
# model = tf.keras.models.load_model(model_file)


## predict data using our model

# use the model to predict the test features
estimated_test_labels = model.predict(test_features)

# compare the predictions to the actual values
plot_true_vs_pred(test_labels*800,estimated_test_labels*800)

# compute the mean absolute error
mape = np.mean( np.abs( (estimated_test_labels.flatten() - test_labels.flatten() )/test_labels.flatten()) ) 
print('The mean absolute percentage error of the trained model is %.4f'%mape)


## uncertainty quantification using dropout nodes

# initialise the architecture
architecture = []

# input layer
architecture.append(tf.keras.layers.Flatten(input_shape=[train_features.shape[1]]))

# first hidden layer
architecture.append(tf.keras.layers.Dense(32, activation='relu'))
architecture.append(tf.keras.layers.Dropout(0.2))
# architecture.append(MCDropout(0.2))

# second hidden layer
architecture.append(tf.keras.layers.Dense(8, activation='relu'))
architecture.append(tf.keras.layers.Dropout(0.1))
#architecture.append(MCDropout(0.1))

# output layer
architecture.append(tf.keras.layers.Dense(1, activation='softplus')) # sigmoid or tanh or softplus

# compile the model
model = tf.keras.Sequential(architecture)
model.compile(loss='mae', optimizer='adam', metrics=['mae','mape','mse','msle'])

# get overview of model architecture
model.summary()

# training the dropout model

# define early stop of training based on validation set
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae',patience=200,restore_best_weights=True)

# run model training and store training history
history = model.fit(train_features,
                    train_labels,
                    epochs=3000,
                    validation_data=(validation_features, validation_labels), 
                    verbose=1,
                    callbacks=[early_stop],
                    batch_size=40) # this is how pieces the data are split into

# plot
plot_training_history(history) 

# produce 100 different predictions for uncertainty
mc_dropout_pred = np.stack([model(test_features,training=True) for i in np.arange(100)])
mc_dropout_mean = mc_dropout_pred.mean(axis=0)
mc_dropout_std = mc_dropout_pred.std(axis=0)

# check shape of data inputs
test_labels.shape
mc_dropout_mean.shape
mc_dropout_std.flatten()

# plot the predictions
fig = plt.figure(figsize=(6, 6))
plt.errorbar(test_labels*800, mc_dropout_mean.flatten()*800, yerr = mc_dropout_std.flatten()*800, fmt='.',alpha=1,ecolor='black',elinewidth=0.5) 
plt.xlim(0,800)
plt.ylim(0,800)
plt.plot([0,800],[0,800],'r-')
plt.xlabel('True diversity')
plt.ylabel('Predicted diversity')
plt.grid()


### Dimensionality reduction

# extract bioclim variables
biome_feature_ids = [i for i,feat_name in enumerate(feature_names) if 'bio_' in feat_name]
biome_features = rescaled_features[:,biome_feature_ids]

# import the umap library
import umap.umap_ as umap

# reduce the dimensions down to fewer axes
reducer = umap.UMAP(n_neighbors=250, min_dist=0)
umap_obj = reducer.fit(biome_features)
biome_features_transformed = reducer.transform(biome_features)
biome_features_transformed.shape

fig = plt.figure(figsize=(8, 6))
plt.scatter(biome_features_transformed[:,0], biome_features_transformed[:,1], c=labels)
plt.colorbar()
plt.grid()
plt.xlabel('UMAP axis 1')
plt.ylabel('UMAP axis 2')
plt.show()

# back transform to 19 dimensions
inv_transf = reducer.inverse_transform(biome_features_transformed)

# use a PCA for dimension reduction
from sklearn.decomposition import PCA

# run the PCA
pca = PCA(n_components=2)
pca.fit(biome_features)
biome_features_pca_transformed = pca.transform(biome_features)

# plot the PCA
fig = plt.figure(figsize=(8, 6))
plt.scatter(biome_features_pca_transformed[:,0], biome_features_pca_transformed[:,1], c=labels)
plt.colorbar()
plt.grid()
plt.xlabel('PCA axis 1')
plt.ylabel('PCA axis 2')
plt.show()

# back transform PCA into original features
inv_transf_pca = pca.inverse_transform(biome_features_pca_transformed)
















