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
    
def build_model(ni, n1, n2, n3, ln, act):

    inputs = keras.layers.Input(shape=(ni), dtype='float32')
    d1 = keras.layers.Dense(n1, input_dim=ni, kernel_regularizer = ln, activation=act)(inputs)
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

def masked_loss_function(y_true, y_pred):
    mask_value=-1
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)

# --------------------------------------------------------------------------------
# builds, compiles and trains a STL_DNN strain model
def run_Strain(x_train, x_test, y_train, y_test, n_inputs, lb, arq, w, act, params_str, strain):
    # L2 regularization
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

    model = build_model(n_inputs, arq[0], arq[1], arq[2],reg, act)
 
    # compile
    model.compile(loss= masked_loss_function, optimizer=adam_opt, metrics = [tf.keras.metrics.Precision(), 
                                                                             tf.keras.metrics.Recall(),
                                                                             tf.keras.metrics.TruePositives(),
                                                                             tf.keras.metrics.TrueNegatives(),
                                                                             tf.keras.metrics.FalsePositives(),
                                                                             tf.keras.metrics.FalseNegatives(),
                                                                             'AUC'])
    # checkpoint
    # adapt paths accordingly
    ckpt_name = './ckpts_models_strains/'+ str(strain) + '/'+ params_str + '.h5'
    ckpt = ModelCheckpoint(ckpt_name, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')

    # train model
    learning_data = model.fit(x = x_train, 
                              y = y_train,
                              epochs = n_epochs,
                              validation_data = (x_test , y_test),
                              callbacks=[early_stop, ckpt],
                              class_weight=w,
                              verbose=0)
    return model
    
#------------------------------------------------------------------------------------------------

#dropout
prob_h1 = 0.25
prob_h2 = 0.15
prob_h3 = 0.1

#batch normalization
momentum_batch_norm = 0.9

#adam optimizador
l_rate = 0.0001

#optimizador
adam_opt = Adam(l_rate)

#early stopping
min_delta_val = 0.0001
patience_val = 500

n_epochs = np.iinfo(np.int32).max

#------------------------------------------------------------------------------------------------
# Weight dict for weighed cost function
pesos = {
    '97':{0:1,1:3.47},
    '98':{0:1,1:1.90},
    '100':{0:1,1:1.56},
    '102':{0:1,1:3.31},
    '1535':{0:1,1:5.09},
    '1537':{0:1,1:5.11},
    '1538':{0:1,1:2.81},
    'Extra':{0:1,1:1.10},
    'Overall':{0:1.40,1:1},
    'Bal': {0:1.1,1:1, -1:0} 
}

# ---- dict containing best hparam combinations for each strain ----   
params_cepas = {'98':{'lb':0.0001,'arq':(200,50,10),'act': 'relu'},
                '100':{'lb':0.001,'arq':(200,50,10),'act': 'relu'},
                '102':{'lb':0.001,'arq':(100,50,15),'act': 'tanh'},
                '1535':{'lb':0.0001,'arq':(10,5,2),'act': 'relu'},
                '1537':{'lb':0.0001,'arq':(10,5,2),'act': 'tanh'}}

# random seeds
seed_lst = [3,7,15,24,45,62,77,79,88,90]
w = None # no weighed cost function

#------------------------------------------------------------------------------------------------

for seed in seed_lst: 
    # for each strain, get hparams an load list of folds
    for key in params_cepas.keys():
        strain = key
        print(key)
        # cargo datos
        list_of_folds = load_data('data.csv', model=strain, stage="5FCV")
        
        # input shape
        n_inputs = list_of_folds[0][0][0].shape[1]
        
        params = params_cepas.get(key)
        lb = params.get('lb')
        arq = params.get('arq')
        w = params.get('w')
        act = params.get('act')
        set_seed(seed)
        is_weighed = 'weighed' if w!=None else 'none'
        
        for idx,split in enumerate(list_of_folds):
            # retrieve data from within the fold
            train,test = split
            x_train, y_train = train
            x_test, y_test = test
            
            params_str = "strain" + str(strain) + "_w_" + str(is_weighed) + "_seed_" + str(seed) + "_fold_" + str(idx) 
            
            # build, compile and train model
            trained_model = run_Strain(x_train, x_test, y_train, y_test, n_inputs, lb, arq, w, act, params_str, strain)
        
            # make predictions
            pred = trained_model.predict(x_test)
            y_pred = np.where((pred>0.5),1,0)
            
            # evaluate and save for future model selection
            # change path names accordingly
            # for monitoring purposes. should be dumped in a log file
            print('Strain '+ str(strain) + ': ' + str(seed) + '_' + is_weighed +'_validacion INTERNA en fold '+str(idx)+': (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score)')
            # ahora tenemos que filtrar los -1 porque estamos usando las etiquetas del consenso
            idxO, y_true, new_y_pred, new_pred = filter_nan(y_test,y_pred,pred)
            print(get_metrics(y_true,new_y_pred))

            name_file = "./logits_strains/"+ str(strain) + "/" + params_str +'.csv'
            df_logits = pd.DataFrame(new_pred, columns = ["logits"])   
            df_logits["y_true"] = y_true
            df_logits["y_pred"] = new_y_pred

            df_logits.to_csv(name_file,index=False)
            K.clear_session()