"""Created on Tue Aug 22 13:52:06 2023"""
import datetime
starttime = datetime.datetime.now() # counter for script running time
print('Started at ', starttime)
import platform
import sys
import keras
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import os
import random
from sklearn.metrics import mean_squared_error
import gsw    
import matplotlib.pyplot as plt
import warnings
import scipy
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from keras.models import Sequential, Model
import matplotlib.patheffects as pe
import cmocean
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import xarray as xr
from sklearn.model_selection import train_test_split



# # Optimizer classes
# Adam = tf.keras.optimizers.Adam

# # Layer classes
# layers = tf.keras.layers

# # Model classes
# Sequential = tf.keras.Sequential
# Model = tf.keras.Model

def pad_to_max_levels(dataarray, max_levels):
    current_levels = dataarray.sizes['N_LEVELS']
    if current_levels < max_levels:
        pad_width = max_levels - current_levels
        
        if np.issubdtype(dataarray.dtype, np.datetime64):
            pad_value = np.datetime64('NaT', 'ns')
        else:
            pad_value = np.nan
        
        padding = xr.DataArray(np.full((dataarray.sizes['N_PROF'], pad_width), pad_value),
                               dims=['N_PROF', 'N_LEVELS'])
        dataarray = xr.concat([dataarray, padding], dim='N_LEVELS')
    return dataarray


def get_traintest_string(test_data_mode, random_seed_train_test_split, reverse=False):
    train_test_split_dict = {'fraction_of_data': 'TTfracdata', 
                             'all_data': 'TTfalse', 
                             'train_test_split': 'TTsplit', 
                             'fraction_of_cruises': 'TTcruises', 
                             'fraction_of_stations': 'TTcasts'}
    
    if reverse:
        train_test_split_dict = {v: k for k, v in train_test_split_dict.items()}
        train_test_split_string = train_test_split_dict[f"TT{test_data_mode}"]
    else:
        train_test_split_string = train_test_split_dict[test_data_mode] + "_seed" + str(random_seed_train_test_split)
    
    return train_test_split_string

def month_to_continuous_curve(month_array):
    angles = (month_array)/12*2*np.pi
    sine = np.sin(angles)
    cos = np.cos(angles)
    return sine, cos
    
def continuous_curve_to_month(sine_array, cos_array):
    # Calculate the angle in radians
    angles = np.arctan2(sine_array, cos_array)
    
    # Convert angle from radians to months (0 to 11)
    month_array = (angles / (2 * np.pi)) * 12
    
    # Handle negative angles to ensure they map correctly to months
    month_array = np.mod(month_array, 12)
    
    # Round to nearest integer to get discrete month values
    month_array = np.round(month_array).astype(int)
    
    return month_array


### Convert lat/lon into their cartesian coordinates
def lat_lon_to_cartesian(lat_deg, lon_deg):
    if (np.max(lon_deg)>360) or (np.min(lon_deg)<0):
        print('Error: lon_deg must be between 0 and 360')
        return
    else:
        # Convert latitude and longitude to radians
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
    
        # Calculate Cartesian coordinates
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
    
        return x, y, z
### Convert cartesian coordinates to lat/lon

def cartesian_to_lat_lon(x, y, z):
    # Calculate latitude in radians
    lat_rad = np.arcsin(z)
    
    # Calculate longitude in radians
    lon_rad = np.arctan2(y, x)
    
    # Convert latitude and longitude from radians to degrees
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    
    # Ensure longitude is within the range [0, 360]
    lon_deg = np.mod(lon_deg, 360)
    
    return lat_deg, lon_deg

def test_on_model_data(model_df, test_data_mode, \
                       test_fraction, random_seed_train_test_split=None):
    if test_data_mode == 'fraction_of_stations':
        profile_labels = model_df['n_profile']
    elif test_data_mode == 'fraction_of_cruises':
        profile_labels = model_df['n_cruise']
    # Calculate the number of unique profiles to keep for evaluation
    unique_stations_fraction = int(np.unique(profile_labels).size * test_fraction)

    # Randomly select rows for evaluation
    rng = np.random.default_rng(seed=random_seed_train_test_split)
    random_indices_eval = rng.choice((np.unique(profile_labels)).astype(int), size=unique_stations_fraction, replace=False)
    indices_kept = np.where(~np.isin(profile_labels, random_indices_eval))[0]
    indices_dropped = np.where(np.isin(profile_labels, random_indices_eval))[0]
    eval_df = model_df.drop(indices_kept)
    dropped_df = model_df.drop(indices_dropped)
    eval_nprof = profile_labels[indices_dropped]
    dropped_nprof = profile_labels[indices_kept]
    
    return dropped_df, eval_df, dropped_nprof, eval_nprof

def binary_space_partitioning(df, dimensions, split_by, max_depth=None, depth = 0):
    if max_depth is None:
        max_depth = len(dimensions) * 2  # You can adjust the max depth as needed
    
    if depth == max_depth or len(df) <= 1:
        return [df]
    
    best_dimension = None
    best_diff = float('inf')
    best_median = None
    best_left_partition = None
    best_right_partition = None

    for dimension in dimensions:
        median = df[dimension].quantile(0.5)
        left_partition = df[df[dimension] <= median]
        right_partition = df[df[dimension] > median]

        # # Below is traditional BSP -- commented out
        # # Calculate the difference in counts between partitions
        # diff = abs(len(left_partition) - len(right_partition))

        # Calculate the difference in unique n_cruise counts between partitions
        if split_by == 'fraction_of_cruises':
            left_unique_cruises = left_partition['n_cruise'].nunique()
            right_unique_cruises = right_partition['n_cruise'].nunique()
        elif split_by == 'fraction_of_stations':
            left_unique_cruises = left_partition['n_profile'].nunique()
            right_unique_cruises = right_partition['n_profile'].nunique()
        elif split_by == 'fraction_of_data':
            left_unique_cruises = left_partition.shape[0]
            right_unique_cruises = right_partition.shape[0]

        diff = abs(left_unique_cruises - right_unique_cruises)
        ## Repeat this until we get the dimension with the smallest diff, i.e., 
        ## the dimension with the most equal partitioning of cruises - that is our dimension of splitting

        if diff < best_diff:
            best_diff = diff
            best_dimension = dimension
            best_median = median
            best_left_partition = left_partition
            best_right_partition = right_partition

    print(best_dimension)
#     # Select dimension to split on
#     dimension = dimensions[depth % len(dimensions)]
    # print('most equal unique n_cruise split = ', best_dimension)
    # Ensure balanced splits
    if len(best_left_partition) == 0 or len(best_right_partition) == 0:
        best_median = df[best_dimension].quantile(0.5 + (depth % 2) * 0.01)
        best_left_partition = df[df[best_dimension] <= best_median]
        best_right_partition = df[df[best_dimension] > best_median]
    
    # Recursively partition the left and right partitions
    return binary_space_partitioning(best_left_partition, dimensions, split_by, max_depth, depth + 1) + \
           binary_space_partitioning(best_right_partition, dimensions, split_by, max_depth, depth + 1)

def test_on_model_data_stratified(df, dimensions, max_depth, split_by, test_size):
    
    # Perform binary space partitioning
    partitions = binary_space_partitioning(df, dimensions, split_by, max_depth, depth = 0)
    
    # Combine partitions into a single DataFrame with a bin label
    df['bin'] = -1
    for i, partition in enumerate(partitions):
        df.loc[partition.index, 'bin'] = i
    
    np.random.seed(42)  # For reproducibility
    eval_split_dim = []
    train_split_dim = []

    if split_by == 'fraction_of_cruises':
        grouped = df.groupby('bin')['n_cruise'].unique().to_dict()
    elif split_by == 'fraction_of_stations':
        grouped = df.groupby('bin')['n_profile'].unique().to_dict()
    elif split_by == 'fraction_of_data':
        grouped = {bin_value: df[df['bin'] == bin_value].index.to_list() for bin_value in df['bin'].unique()}
    
    # Subsample the dictionary
    subsampled_data = {}
    for key, values in grouped.items():
        num_to_remove = int(len(values) * test_size)
        values_list = list(values)
        indices_to_remove = random.sample(range(len(values_list)), num_to_remove)
        subsampled_values = [v for i, v in enumerate(values_list) if i not in indices_to_remove]
        subsampled_data[key] = subsampled_values
    
    # Filter the DataFrame based on the subsampled dictionary
    if split_by == 'fraction_of_cruises':
        filtered_df = pd.concat([
            df[(df['bin'] == bin_value) & (df['n_cruise'].isin(cruises))]
            for bin_value, cruises in subsampled_data.items()
        ])
        removed_df = pd.concat([
            df[(df['bin'] == bin_value) & (~df['n_cruise'].isin(cruises))]
            for bin_value, cruises in subsampled_data.items()
        ])
    
    elif split_by == 'fraction_of_stations':
        filtered_df = pd.concat([
            df[(df['bin'] == bin_value) & (df['n_profile'].isin(cruises))]
            for bin_value, cruises in subsampled_data.items()
        ])
        removed_df = pd.concat([
            df[(df['bin'] == bin_value) & (~df['n_profile'].isin(cruises))]
            for bin_value, cruises in subsampled_data.items()
        ])
    
    elif split_by == 'fraction_of_data':
        for bin_value, split_dim in grouped.items():
            split_dim = np.array(split_dim)
            np.random.shuffle(split_dim)
            split_index = int((len(split_dim) * test_size))
            eval_split_dim.extend(split_dim[:split_index])
            train_split_dim.extend(split_dim[split_index:])
    
            eval_df = df.loc[eval_split_dim]
            model_df = df.loc[train_split_dim]
    
    if split_by == 'fraction_of_cruises' or split_by == 'fraction_of_stations':
        # Reset index if desired
        filtered_df.reset_index(drop=True, inplace=True)    
        # Reset index for removed_df if desired
        removed_df.reset_index(drop=True, inplace=True)

    
        eval_df = removed_df
        model_df = filtered_df

    return eval_df, model_df

def get_variables_for_training(dataframe, predicted_variable, predictor_variables=None):
    # dataframe = dataframe[["year", "time_of_year_function", "latitude", "longitude", "depth", "temperature", 
    #                    "salinity", predicted_variable]]
        
    if predictor_variables is None:
        dataframe = dataframe[["year", "month_sin", "month_cos", "location_X", \
                               "location_Y", "location_Z", "depth", "temperature",\
                               "bottom_depth", "salinity", predicted_variable]]
    else:
        dataframe = dataframe[predictor_variables + [predicted_variable]]
        
    return dataframe

def get_train_test_split(glodap_df, glodap_eval_df, predicted_variable, predictor_variables, test_fraction):
    
    # choose all columns used for training (including the predicted variable)
    glodap_df = get_variables_for_training(glodap_df, predicted_variable, predictor_variables)
    glodap_eval_df = get_variables_for_training(glodap_eval_df, predicted_variable, predictor_variables)

    x_train = np.array(glodap_df.drop(predicted_variable,axis=1))
    y_train = np.array(glodap_df[predicted_variable]).T # series with one column need to be transposed, because they will end up having the shape 1,n instead of n,1
    
    x_eval = np.array(glodap_eval_df.drop(predicted_variable,axis=1))
    y_eval = np.array(glodap_eval_df[predicted_variable]).T # series with one column need to be transposed, because they will end up having the shape 1,n instead of n,1

    return x_train, x_eval, y_train, y_eval

def get_out_of_sample_training(glodap_df, predicted_variable, predictor_variables):
    
    # choose all columns used for training (including the predicted variable)
    glodap_df = get_variables_for_training(glodap_df, predicted_variable, predictor_variables)

    x_train = np.array(glodap_df.drop(predicted_variable,axis=1))
    y_train = np.array(glodap_df[predicted_variable]).T # series with one column need to be transposed, because they will end up having the shape 1,n instead of n,1
    
    return x_train, y_train


def set_up_callbacks(model_run_string, sample_log_directory, checkpoint_file_directory, checkpointer):
    ''' CSVLogger
    *Callback that streams epoch results to a CSV file.* (https://keras.io/api/callbacks/csv_logger/) '''
    starttime_logger = datetime.datetime.now()
    csv_logger = CSVLogger(sample_log_directory + model_run_string + '_sample_log' + '.csv', 
                           append=True, separator=',')
    ''' EarlyStopping
    Stop training when a monitored metric has stopped improving (https://keras.io/api/callbacks/early_stopping/)'''
    earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1) # the monitor argument is probably 'loss' by default

    
    lr_reduction_factor = 0.2
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_reduction_factor,
                              patience=5, min_lr=0.0000001, verbose=1, cooldown=1)
    
    if checkpointer:
        ''' ModelCheckpoint
        *Callback to save the Keras model or model weights at some frequency.* (https://keras.io/api/callbacks/model_checkpoint/)'''
        checkpointer = ModelCheckpoint(checkpoint_file_directory + 'checkpt_{val_loss:.2f}_p1_sample.h5', 
                                       verbose=1, save_best_only=True)
        
        callbacks_list = [csv_logger, earlystopper, checkpointer, reduce_lr]
    else:
        callbacks_list = [csv_logger, earlystopper, reduce_lr]
    
    return callbacks_list

def build_functional_fore(x_data_shape, learning_rate, feature_weights=None):
            
    input_shape = x_data_shape[1:]
    
    #############################################
    # Create Keras model

    input_layer = layers.Input(shape=input_shape)
    
    if feature_weights is not None:
        # Multiply each feature by its weight using element-wise multiplication
        input_layer = (layers.Multiply()([input_layer, feature_weights]))
        
    return input_layer

def build_seq_aft(model_run_string, model, learning_rate, sample_log_directory, checkpoint_file_directory, checkpointer=False):
    
    ''' Create Adam optimizer
    Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
    According to Kingma et al., 2014 (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) '''
    opt = Adam(learning_rate=learning_rate) 
    # Available optimizers
    # SGD (Stochastic Gradient Descent), RMSprop, Adagrad, Adam , Adamax, Nadam
    ''' Set up callbacks '''
    callbacks_list = set_up_callbacks(model_run_string, sample_log_directory, checkpoint_file_directory, checkpointer) 
    
    # from zemskova.functions import r_score, mse_nonzero

    # loss_function = tf.keras.losses.MeanSquaredError() # 'mse'
    # loss_function = mse_nonzero # 'mse'
    loss_function = 'mse' # 'mse'

    # loss_function = tf.keras.metrics.MeanSquaredError() # 'mse'
    # metrics = [tf.keras.metrics.MeanAbsoluteError()]
    # metrics = [r_score, mse_nonzero]
    # from tensorflow_addons.metrics import RSquare
    # metrics = [tf.keras.metrics.R2Score(), tf.keras.metrics.RootMeanSquaredError()]
    metrics = [tf.keras.metrics.RootMeanSquaredError()]

    '''Configures the model for training.'''
    model.compile(
        optimizer=opt, # String (name of optimizer) or optimizer instance.
        loss=loss_function, # Loss function. May be a string (name of loss function), or a tf.keras.losses.Loss instance.
        metrics=metrics, # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
        loss_weights=None, # Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. 
        weighted_metrics=None, # List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
        run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function. 
        steps_per_execution=None, # Int. Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
        jit_compile=None # If True, compile the model training step with XLA. XLA is an optimizing compiler for machine learning. jit_compile is not enabled for by default. This option cannot be enabled with run_eagerly=True. 
        )

    return model, callbacks_list

def build_func_residual_connections(model_run_string, x_data_shape, learning_rate, sample_log_directory, checkpoint_file_directory, checkpointer=False, feature_weights=None):
    
    input_layer = build_functional_fore(x_data_shape, learning_rate, feature_weights=None)

    x = input_layer
    
    # Projection layer to match input and output dimensions
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # Add a series of dense layers with residual connections
    for _ in range(3):  # You can adjust the number of layers
        residual = x
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Add()([x, residual])  # Residual connection


    # Output layer for regression
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    
    model, callbacks_list = build_seq_aft(model_run_string, model, learning_rate, sample_log_directory, checkpoint_file_directory, checkpointer=False)


    return model, callbacks_list

def call_build_model_based_on_model_name(model_run_string, model_name, x_data_shape, learning_rate, 
                                       sample_log_directory, checkpoint_file_directory, 
                                       checkpointer, feature_weights):
    
    if model_name == 'FFN1':
        model, callbacks_list = build_seq_FFN1(model_run_string, x_data_shape, learning_rate, 
                                               sample_log_directory, checkpoint_file_directory, 
                                               checkpointer=False, feature_weights=feature_weights)
    elif model_name == 'LSTM2':
        model, callbacks_list = build_seq_LSTM2(model_run_string, x_data_shape, learning_rate, 
                                               sample_log_directory, checkpoint_file_directory,
                                               checkpointer=False, feature_weights=feature_weights)
    elif model_name == 'LSTM3':
        model, callbacks_list = build_seq_LSTM3(model_run_string, x_data_shape, learning_rate, 
                                               sample_log_directory, checkpoint_file_directory,
                                               checkpointer=False, feature_weights=feature_weights)
    elif model_name == 'LSTM4':
        model, callbacks_list = build_seq_LSTM4(model_run_string, x_data_shape, learning_rate, 
                                               sample_log_directory, checkpoint_file_directory,
                                               checkpointer=False, feature_weights=feature_weights)
    elif model_name == 'LSTM5':
        model, callbacks_list = build_seq_LSTM5(model_run_string, x_data_shape, learning_rate, 
                                               sample_log_directory, checkpoint_file_directory,
                                               checkpointer=False, feature_weights=feature_weights)
    elif model_name == 'LSTM6':
        model, callbacks_list = build_seq_LSTM6(model_run_string, x_data_shape, learning_rate, 
                                               sample_log_directory, checkpoint_file_directory,
                                               checkpointer=False, feature_weights=feature_weights)
    elif model_name == 'LSTM7':
        model, callbacks_list = build_seq_LSTM7(model_run_string, x_data_shape, learning_rate, 
                                               sample_log_directory, checkpoint_file_directory,
                                               checkpointer=False, feature_weights=feature_weights)
    elif model_name == 'LSTM8':
        model, callbacks_list = build_func_LSTM8(model_run_string, x_data_shape, learning_rate, 
                                               sample_log_directory, checkpoint_file_directory,
                                               checkpointer=False, feature_weights=feature_weights)
    elif model_name == 'attn':
        model, callbacks_list = build_func_attention(model_run_string, x_data_shape, learning_rate, 
                                               sample_log_directory, checkpoint_file_directory,
                                               checkpointer=False, feature_weights=feature_weights)
    elif model_name == 'resnet':
        model, callbacks_list = build_func_residual_connections(model_run_string, x_data_shape, learning_rate, 
                                               sample_log_directory, checkpoint_file_directory,
                                               checkpointer=False, feature_weights=feature_weights)
    elif model_name == 'wide_deep':
        model, callbacks_list = build_func_wide_deep(model_run_string, x_data_shape, learning_rate, 
                                               sample_log_directory, checkpoint_file_directory,
                                               checkpointer=False, feature_weights=feature_weights)

    return model, callbacks_list

def train_FFN(model, x_train, y_train, x_eval, y_eval, EPOCHS, callbacks_list, batch_size=None, verbose='auto', initial_epoch=0):
    # You can now iterate on your training data in batches:
    
        
    
    ''' Trains the model for a fixed number of epochs (iterations on a dataset).'''
    history = model.fit(
        x=x_train, y=y_train, 
        validation_data=(x_eval, y_eval),
        epochs=EPOCHS, # Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided (unless the steps_per_epoch flag is set to something other than None). 
        batch_size=batch_size, # Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances (since they generate batches).
        verbose=verbose, # verbose: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
        initial_epoch=initial_epoch, # Integer. Epoch at which to start training (useful for resuming a previous training run).
        steps_per_epoch=None, # Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
        # validation_split=0.1, # Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. 
        # validation_data=(x_eval, y_eval), # THIS MEANS THAT THERE WON'T BE val_loss-values - WHY? This can be either - a generator or a `Sequence` object for the validation data - tuple `(x_val, y_val)` - tuple `(x_val, y_val, val_sample_weights)`
        callbacks=callbacks_list)
    
    # Evaluate your test loss and metrics in one line:
    
    loss_and_metrics = model.evaluate(x_eval, y_eval, batch_size=128)
    
    return model, history