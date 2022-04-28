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

def masked_loss_function(y_true, y_pred):
    mask_value=-1
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)

# --------------------------------------------------------------------------------
# Load data (fixed partitions for grid search)
data_path = 'data.csv'
train, internal = load_data(data_path, model="MTL", stage="GS")
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

# Target-specific core of the MTL_DNN architecture
# Two-layer option
def two_specific_layers(x, n2, n3):
    # STRAIN 1
    y1 = Dense(n2, activation=act)(x)
    y1 = BatchNormalization(momentum=momentum_batch_norm)(y1)
    y1 = Dropout(prob_h2)(y1)
    y1 = Dense(n3, activation=act)(y1)
    y1 = BatchNormalization(momentum=momentum_batch_norm)(y1)
    y1 = Dropout(prob_h3)(y1)

    # STRAIN 2
    y2 = Dense(n2, activation=act)(x)
    y2 = BatchNormalization(momentum=momentum_batch_norm)(y2)
    y2 = Dropout(prob_h2)(y2)
    y2 = Dense(n3, activation=act)(y2)
    y2 = BatchNormalization(momentum=momentum_batch_norm)(y2)
    y2 = Dropout(prob_h3)(y2)
    
    # STRAIN 3
    y3 = Dense(n2, activation=act)(x)
    y3 = BatchNormalization(momentum=momentum_batch_norm)(y3)
    y3 = Dropout(prob_h2)(y3)
    y3 = Dense(n3, activation=act)(y3)
    y3 = BatchNormalization(momentum=momentum_batch_norm)(y3)
    y3 = Dropout(prob_h3)(y3)

    # STRAIN 4
    y4 = Dense(n2, activation=act)(x)
    y4 = BatchNormalization(momentum=momentum_batch_norm)(y4)
    y4 = Dropout(prob_h2)(y4)
    y4 = Dense(n3, activation=act)(y4)
    y4 = BatchNormalization(momentum=momentum_batch_norm)(y4)
    y4 = Dropout(prob_h3)(y4)
    
    # STRAIN 5
    y5 = Dense(n2, activation=act)(x)
    y5 = BatchNormalization(momentum=momentum_batch_norm)(y5)
    y5 = Dropout(prob_h2)(y5)
    y5 = Dense(n3, activation=act)(y5)
    y5 = BatchNormalization(momentum=momentum_batch_norm)(y5)
    y5 = Dropout(prob_h3)(y5)
    
    return y1,y2,y3,y4,y5

# Target-specific core of the MTL_DNN architecture
# One-layer option
def one_specific_layers(x, n2):
    # STRAIN 1
    y1 = Dense(n2, activation=act)(x)
    y1 = BatchNormalization(momentum=momentum_batch_norm)(y1)
    y1 = Dropout(prob_h2)(y1)

    # STRAIN 2
    y2 = Dense(n2, activation=act)(x)
    y2 = BatchNormalization(momentum=momentum_batch_norm)(y2)
    y2 = Dropout(prob_h2)(y2)
    
    # STRAIN 3
    y3 = Dense(n2, activation=act)(x)
    y3 = BatchNormalization(momentum=momentum_batch_norm)(y3)
    y3 = Dropout(prob_h2)(y3)

    # STRAIN 4
    y4 = Dense(n2, activation=act)(x)
    y4 = BatchNormalization(momentum=momentum_batch_norm)(y4)
    y4 = Dropout(prob_h2)(y4)
    
    # STRAIN 5
    y5 = Dense(n2, activation=act)(x)
    y5 = BatchNormalization(momentum=momentum_batch_norm)(y5)
    y5 = Dropout(prob_h2)(y5)
    
    return y1,y2,y3,y4,y5

# --------------------------------------------------------------------------------

# General MTL_DNN architecture
def build_model(ni, n0, n1, n2, n3, ln, act, spec_layers):

    # Shared core
    model_input = Input(shape=(ni,))
    x = Dense(n0, activation=act)(model_input)
    x = BatchNormalization(momentum=momentum_batch_norm)(x)
    x = Dropout(prob_h1)(x)
    x = Dense(n1, activation=act)(x)
    x = BatchNormalization(momentum=momentum_batch_norm)(x)
    x = Dropout(prob_h2)(x)
    x = Dense(n2, activation=act)(x)
    x = BatchNormalization(momentum=momentum_batch_norm)(x)
    x = Dropout(prob_h3)(x)
    
    # Target-specific core
    if (spec_layers==0):
        y1 = Dense(units=1, activation='sigmoid', name='output_1')(x)
        y2 = Dense(units=1, activation='sigmoid', name='output_2')(x)
        y3 = Dense(units=1, activation='sigmoid', name='output_3')(x)
        y4 = Dense(units=1, activation='sigmoid', name='output_4')(x)
        y5 = Dense(units=1, activation='sigmoid', name='output_5')(x)
    else:
        if (spec_layers==1):
            y1, y2, y3, y4, y5 = one_specific_layers(x, n2)
        else:
            if (spec_layers==2):
                y1, y2, y3, y4, y5 = two_specific_layers(x, n2, n3)

        # Outputs 1 to 5
        y1 = Dense(units=1, activation='sigmoid', name='output_1')(y1)
        y2 = Dense(units=1, activation='sigmoid', name='output_2')(y2)
        y3 = Dense(units=1, activation='sigmoid', name='output_3')(y3)
        y4 = Dense(units=1, activation='sigmoid', name='output_4')(y4)
        y5 = Dense(units=1, activation='sigmoid', name='output_5')(y5)

    model = Model(inputs=model_input, outputs=[y1, y2, y3, y4, y5])
    return model

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
pesos_5_cepas = {'output_1':pesos.get('98'), 'output_2':pesos.get('100'), 'output_3':pesos.get('102'), 'output_4':pesos.get('1535'), 'output_5':pesos.get('1537')}
pesos_bal = {'output_1':pesos.get('Bal'), 'output_2':pesos.get('Bal'), 'output_3':pesos.get('Bal'), 'output_4':pesos.get('Bal'), 'output_5':pesos.get('Bal')}
class_weight_lst = [None, pesos_5_cepas, pesos_bal]

# Number of layers in the target-specific core
spec_layers_lst = [0,1,2]

# Size of input layer
n_inputs = X_train.shape[1]

# Num epochs
n_epochs = np.iinfo(np.int32).max

# Different random seeds for parameter initialization
seed_lst = [0,20,25,65,92,39,7,88,23,55,29,5,15,30,44,70,10,18,69,80]

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
    
        np.random.shuffle(spec_layers_lst)
        spec_lay  = random.choice(spec_layers_lst)
    
        params ='lb_'+str(lb)+'_spec_lay_'+str(spec_lay)+'_n0_'+str(n0)+'_n1_'+str(n1)+'_n2_'+str(n2)+'_n3_'+str(n3)+'_w_'+str(w)
        
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
    model = build_model(n_inputs, n0, n1, n2, n3, reg, act,spec_lay)

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
    y_pred = model.predict(X_internal)

    # evaluate and save for future model selection
    # change path names accordingly
    y_pred_98 = np.where(y_pred[0] > 0.5, 1,0)
    y_pred_100 = np.where(y_pred[1] > 0.5, 1,0)
    y_pred_102 = np.where(y_pred[2] > 0.5, 1,0)
    y_pred_1535 = np.where(y_pred[3] > 0.5, 1,0)
    y_pred_1537 = np.where(y_pred[4] > 0.5, 1,0)

    # for monitoring purposes. should be dumped in a log file
    print("Strain TA98")
    print('internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_internal[0],y_pred_98,y_pred[0])
    print(get_metrics(new_real,new_y_pred,new_prob))

    print("Strain TA100")
    print('internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_internal[1],y_pred_100,y_pred[1])
    print(get_metrics(new_real,new_y_pred,new_prob))
    
    print("Strain TA102")
    print('internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_internal[2],y_pred_102,y_pred[2])
    print(get_metrics(new_real,new_y_pred,new_prob))

    print("Strain TA1535")
    print('internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_internal[3],y_pred_1535,y_pred[3])
    print(get_metrics(new_real,new_y_pred,new_prob))

    print("Strain TA1537")
    print('internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_internal[4],y_pred_1537,y_pred[4])
    print(get_metrics(new_real,new_y_pred,new_prob))
    
    sys.stdout.flush()

    #limpiar memoria
    K.clear_session()

# --------------------------------------------------------------------------------
