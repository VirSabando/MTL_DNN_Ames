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
# Load data (five folds for 5FCV)
data_path = 'data.csv'
list_of_folds = load_data(data_path, model="MTL", stage="5FCV")
# --------------------------------------------------------------------------------

# Some constants

# learning rate for optimizer
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

# Size of input layer
n_inputs = list_of_folds[0][0][0].shape[1]

# Num epochs
n_epochs = np.iinfo(np.int32).max

# activation fn
act = 'relu'

#------------------------------------------------------------------------------------
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
    'Overall':{0:1.40, 1:1, -1:0}
}

# Cost weighing alternatives
pesos_5_cepas = {'output_1':pesos.get('98'), 'output_2':pesos.get('100'), 'output_3':pesos.get('102'), 'output_4':pesos.get('1535'), 'output_5':pesos.get('1537')}

# ---- dics containing best hparam combinations, using both weighed and non-weighed cost functions ----
params_dict = {'a':{'lb':0.005,'spec_lay':0,'arq':(100,50,10,5),'w': None},
              'b':{'lb':0.005,'spec_lay':2,'arq':(200,100,50,10),'w': pesos_5_cepas}}

# random seeds
seed_lst = [3,7,15,24,45,62,77,79,88,90]

#------------------------------------------------------------------------------------

# builds, compiles and trains a MTL_DNN model
def run_MTL(x_train, x_test, y_train, y_test, lb, spec_lay, arq, w, fold):
    
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

    model = build_model(n_inputs, arq[0], arq[1], arq[2], arq[3], reg, act, spec_lay)

    # compile model
    model.compile(loss= masked_loss_function, optimizer=adam_opt, metrics = [tf.keras.metrics.Precision(), 
                                                                             tf.keras.metrics.Recall(),
                                                                             tf.keras.metrics.TruePositives(), 
                                                                             tf.keras.metrics.TrueNegatives(), 
                                                                             tf.keras.metrics.FalsePositives(), 
                                                                             tf.keras.metrics.FalseNegatives(),'AUC'])
    # checkpoint
    # adapt paths accordingly
    ckpt_name_cons = './ckpts_5fcv/5cepas/5_cepas'+ "_Params_" + str(params) + "_seed_" + str(seed) + "_fold_" + str(fold)+ '.h5'
    ckpt_cons = ModelCheckpoint(ckpt_name_cons, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
    
    # train the model
    learning_data = model.fit(x=x_train,
                              y=y_train,
                              epochs=n_epochs,
                              validation_data=(x_test,y_test),
                              callbacks=[early_stop],
                              class_weight=w,
                              verbose=0)
    
    return model

#------------------------------------------------------------------------------------
for seed in seed_lst:
    for idx,split in enumerate(list_of_folds):
        
        # retrieve data from within the fold
        train,test = split
        x_train, y_train = train
        x_test, y_test = test
        
        # get params
        for key in params_dict.keys():
            params = params_dict.get(key)
            lb = params.get('lb')
            arq = params.get('arq')
            w = params.get('w')
            spec_lay = params.get('spec_lay')
            set_seed(seed)
        
            # build, compile and train model
            trained_model = run_MTL(x_train, x_test, y_train, y_test, lb, spec_lay, arq, w, idx)
            
            # make predictions
            pred = trained_model.predict(x_test)
            
            y_pred_98 = np.where(pred[0] > 0.5, 1,0)
            y_pred_100 = np.where(pred[1] > 0.5, 1,0)
            y_pred_102 = np.where(pred[2] > 0.5, 1,0)
            y_pred_1535 = np.where(pred[3] > 0.5, 1,0)
            y_pred_1537 = np.where(pred[4] > 0.5, 1,0)
    
            is_weighed = 'weighed' if w!=None else 'none'

            # evaluate and save for future model selection
            # change path names accordingly
            # for monitoring purposes. should be dumped in a log file
            print("Strain TA98")
            print(is_weighed + '_'+ str(seed)+'_validacion interna en fold '+str(idx)+': (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score)')
            idx98, y_true_98, new_y_pred_98, new_pred_98 = filter_nan(y_test[0],y_pred_98,pred[0])
            print(get_metrics(y_true_98,new_y_pred_98))

            print("Strain TA100")
            print(is_weighed + '_'+ str(seed)+'_validacion interna en fold '+str(idx)+': (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score)')
            idx100, y_true_100, new_y_pred_100, new_pred_100 = filter_nan(y_test[1],y_pred_100,pred[1])
            print(get_metrics(y_true_100,new_y_pred_100))
            
            print("Strain TA102")
            print(is_weighed + '_'+ str(seed)+'_validacion interna en fold '+str(idx)+': (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score)')
            idx102, y_true_102, new_y_pred_102, new_pred_102 = filter_nan(y_test[2],y_pred_102,pred[2])
            print(get_metrics(y_true_102,new_y_pred_102))

            print("Strain TA1535")
            print(is_weighed + '_'+ str(seed)+'_validacion interna en fold '+str(idx)+': (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score)')
            idx1535, y_true_1535, new_y_pred_1535, new_pred_1535 = filter_nan(y_test[3],y_pred_1535,pred[3])
            print(get_metrics(y_true_1535,new_y_pred_1535))

            print("Strain TA1537")
            print(is_weighed + '_'+ str(seed)+'_validacion interna en fold '+str(idx)+': (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score)')
            idx1537, y_true_1537, new_y_pred_1537, new_pred_1537 = filter_nan(y_test[4],y_pred_1537,pred[4])
            print(get_metrics(y_true_1537,new_y_pred_1537))
                
            sys.stdout.flush()
  
            K.clear_session()
    
            # Optional: make csv files with ground truth labels, predictions and logits for future inspection
            # change paths accordingly
            name_file = "./logits_MTL/seed_" + str(seed) + '_fold_' + str(idx) + '_w_'+ is_weighed +'.csv'
            df = pd.DataFrame(new_pred_98, columns = ["logits_98"], index=idx98)
            df2 = pd.DataFrame(new_pred_100, columns = ["logits_100"], index = idx100)
            dfa = pd.DataFrame(new_pred_102, columns = ["logits_102"], index = idx102)
            df3 = pd.DataFrame(new_pred_1535, columns = ["logits_1535"], index = idx1535)
            df4 = pd.DataFrame(new_pred_1537, columns = ["logits_1537"], index = idx1537)
            
            df5 = pd.DataFrame(y_true_98, columns = ["y_true_98"], index=idx98)
            df6 = pd.DataFrame(y_true_100, columns = ["y_true_100"], index = idx100)
            dfb = pd.DataFrame(y_true_102, columns = ["y_true_102"], index = idx102)
            df7 = pd.DataFrame(y_true_1535, columns = ["y_true_1535"], index = idx1535)
            df8 = pd.DataFrame(y_true_1537, columns = ["y_true_1537"], index = idx1537)
            
            df9 = pd.DataFrame(new_y_pred_98, columns = ["y_pred_98"], index=idx98)
            df10 = pd.DataFrame(new_y_pred_100, columns = ["y_pred_100"], index = idx100)
            dfc = pd.DataFrame(new_y_pred_102, columns = ["y_pred_102"], index = idx102)
            df11 = pd.DataFrame(new_y_pred_1535, columns = ["y_pred_1535"], index = idx1535)
            df12 = pd.DataFrame(new_y_pred_1537, columns = ["y_pred_1537"], index = idx1537)
            
            whole_idx = pd.DataFrame(index=np.arange(len(y_test[0])))
            
            new = pd.concat([df,df2,dfa,df3,df4,df5,df6,dfb,df7,df8,df9,df10,dfc,df11,df12,whole_idx], axis=1, join='outer') 
            new = new.fillna(-1)

            new.to_csv(name_file,index=False)