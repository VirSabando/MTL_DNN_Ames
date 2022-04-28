# --------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import utils
from tensorflow.keras import losses
from tensorflow.keras import backend as K

import numpy as np 
import pandas as pd
import random

import h5py
import sys

from compute_metrics import *
from data import load_data

tf.compat.v1.disable_eager_execution()
# --------------------------------------------------------------------------------

def set_seed(s):
    K.clear_session()
    
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value= s

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value)

# --------------------------------------------------------------------------------
# This function is not actually necessary in single-task models if you remove the
# compounds labeled as undefined (-1) first, but we did not :)
def masked_loss_function(y_true, y_pred):
    mask_value=-1
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)

# --------------------------------------------------------------------------------
# Load data (fixed partitions for grid search)
data_path = 'data.csv'
train, internal = load_data(data_path, model="Overall", stage="GS")
X_train, y_train = train
X_internal, y_internal = internal
# --------------------------------------------------------------------------------

# Some constants

#adam optimizador
l_rate = 0.0001

#dropout
prob_h1 = 0.25
prob_h2 = 0.15
prob_h3 = 0.1

#batch normalization
momentum_batch_norm = 0.9

#early stopping
min_delta_val = 0.0005
patience_val = 1000

# --------------------------------------------------------------------------------

# Hyperparameters to try at the grid search
act = 'relu'

# L2 regularization lambda
lb_lst = [0.001,0.005,0.01]

# Number of units in the shared core and target-specific core
n_lst = [(100,50,10,5), (100,50,20,10), (200,100,50,10), (200,100,20,10), (200,100,10,5)]

# Weight dict for weighed cost function
pesos = {
    '97':{0:1, 1:3.47, -1:0},
    '98':{0:1, 1:1.90, -1:0},
    '100':{0:1, 1:1.56, -1:0},
    '102':{0:1, 1:3.31, -1:0},
    '1535':{0:1, 1:5.09, -1:0},
    '1537':{0:1, 1:5.11, -1:0},
    '1538':{0:1, 1:2.81, -1:0},
    'Extra':{0:1, 1:1.10, -1:0},
    'Overall':{0:1.40, 1:1, -1:0},
    'Bal':{0:1.1, 1:1, -1:0}
}

# Cost weighing alternatives
class_weight_lst = [None, pesos.get('Overall'), pesos.get('Bal')]

# Size of input layer
n_inputs = X_train.shape[1]

# Num epochs
n_epochs = np.iinfo(np.int32).max

# Different random seeds for parameter initialization
seed_lst = [0,20,25,65,92,39,7,88,23,55,29,5,15,30,44,70,10,18,69,80]

# --------------------------------------------------------------------------------
    
# General STL_DNN architecture
def build_model(ni, n0, n1, n2, n3, ln, act):

    inputs = keras.layers.Input(shape=(ni), dtype='float32')
    d0 = keras.layers.Dense(n0, input_dim=ni, kernel_regularizer = ln, activation=act)(inputs)
    bn0 = keras.layers.BatchNormalization(momentum=momentum_batch_norm)(d0)
    dr0 = keras.layers.Dropout(prob_h1)(bn0)
    
    d1 = keras.layers.Dense(n1, kernel_regularizer = ln, activation=act)(dr0)
    bn1 = keras.layers.BatchNormalization(momentum=momentum_batch_norm)(d1)
    dr1 = keras.layers.Dropout(prob_h1)(bn1)

    d2 = keras.layers.Dense(n2,kernel_regularizer = ln, activation=act)(dr1)
    bn2 = keras.layers.BatchNormalization(momentum=momentum_batch_norm)(d2)
    dr2 = keras.layers.Dropout(prob_h2)(bn2)

    d3 = keras.layers.Dense(n3,kernel_regularizer = ln, activation=act)(dr2)
    bn3 = keras.layers.BatchNormalization(momentum=momentum_batch_norm)(d3)
    dr3 = keras.layers.Dropout(prob_h2)(bn3)

    oo = keras.layers.Dense(1,activation='sigmoid')(dr3)
    model = keras.models.Model(inputs=inputs, outputs=oo)
    return model
# --------------------------------------------------------------------------------

# set of already tested param combinations, to carry out a random grid search
tested_params = set()

# adapt range to number of random initializations
for i in range(20):
    while(True):
        # Shuffle of param lists and random selection of new param combinations
        np.random.shuffle(lb_lst)
        lb = random.choice(lb_lst)
    
        np.random.shuffle(n_lst)
        n0, n1, n2, n3 = random.choice(n_lst)
    
        np.random.shuffle(class_weight_lst)
        w = random.choice(class_weight_lst)
    
        params ='lb_'+str(lb)+'_n0_'+str(n0)+'_n1_'+str(n1)+'_n2_'+str(n2)+'_n3_'+str(n3)+'_w_'+str(w)
        
        if params not in tested_params:
            tested_params.add(params)
            break
    
    # for monitoring purposes
    params = 'Run_' + str(i) + '_' + params
    print(params)
    sys.stdout.flush()
    
    # random seed
    seed_r = seed_lst[i]
    set_seed(seed_r)
    
    # regularization
    reg = l2(lb)
    # optimizer
    adam_opt = Adam(l_rate)
    # early stopping
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=min_delta_val, 
                               patience=patience_val, 
                               verbose=0, 
                               mode='min',
                               restore_best_weights=True)

    # build MTL_DNN model
    model = build_model(n_inputs, n0, n1, n2, n3, reg, act)

    # compile model
    model.compile(loss= masked_loss_function, optimizer=adam_opt, metrics = [tf.keras.metrics.Precision(),
                                                                             tf.keras.metrics.Recall(),
                                                                             tf.keras.metrics.TruePositives(),
                                                                             tf.keras.metrics.TrueNegatives(),
                                                                             tf.keras.metrics.FalsePositives(),
                                                                             tf.keras.metrics.FalseNegatives()])

    # train model
    learning_data = model.fit(x=X_train, y=y_train,
                              epochs=n_epochs,
                              validation_data=(X_internal,y_internal),
                              callbacks=[early_stop],
                              class_weight=w,
                              verbose=0)

    # make predictions
    pred = model.predict(X_internal)
    y_pred = np.where((pred>0.5),1,0)

    # evaluate and save for future model selection
    # change path names accordingly
    # for monitoring purposes. should be dumped in a log file
    print("Overall")
    print('internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, y_true, y_pred, pred = filter_nan(y_test,y_pred,pred)
    print(get_metrics(y_true,y_pred))
    
    sys.stdout.flush()

    #limpiar memoria
    K.clear_session()

# --------------------------------------------------------------------------------
