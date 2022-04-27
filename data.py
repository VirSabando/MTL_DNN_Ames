import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

SEED = 202042

def load_data(data_path, model, stage):
    # load data
    df = pd.read_csv(data_path)

    # PARTITIONS
    train = df.loc[df['Partition'].str.contains('Train')]
    internal = df.loc[df['Partition'].str.contains('Internal')]
    external = df.loc[df['Partition'].str.contains('External')]

    # Select all columns corresponding to molecular descriptors
    # Discard those descriptors showing constant values
    X_train = train.values[:,:-8]
    l = [not np.nanmax(c)== np.nanmin(c) for c in X_train.T]
    X_internal = internal.values[:,:-8]
    l = l or [not np.nanmax(c)== np.nanmin(c) for c in X_internal.T]
    X_external = external.values[:,:-8]
    l = l or[not np.nanmax(c)== np.nanmin(c) for c in X_external.T]
    X_train = X_train[:,l]
    X_internal = X_internal[:,l]
    X_external = X_external[:,l]

    # Target values per partition - MTL
    y_train_MTL = train[['TA98','TA100', 'TA102','TA1535','TA1537']]
    y_internal_MTL = internal[['TA98','TA100', 'TA102','TA1535','TA1537']]
    y_external_MTL = external[['TA98','TA100', 'TA102','TA1535','TA1537']]
    
    # Target values per partition - Overall
    y_train_overall = train['Overall']
    y_internal_overall = internal['Overall']
    y_external_overall = external['Overall']

    output = ()
    
    # If performing five-fold CV, merge train and internal partitions
    if (stage=="5FCV"):
        X = np.concatenate((X_train, X_internal), axis=0)

        # Individual targets for MTL model
        y = pd.concat([y_train_MTL, y_internal_MTL])
        y_MTL = [y.iloc[:,0].values, 
                 y.iloc[:,1].values, 
                 y.iloc[:,2].values, 
                 y.iloc[:,3].values, 
                 y.iloc[:,4].values]

        # Overall target for consensus and STL_DNN_overall model    
        y_overall = pd.concat([y_train_overall, y_internal_overall]) 
        y_overall = y_overall.values
    
        # CFV
        num_folds = 5
        
        # In this case, the output is a list of tuples containing the training folds
        output = []
    
        labels_dict = {
            'MTL':y_MTL,
            'Overall':y_overall      
        }
    
        features = X
        labels = labels_dict.get(model)  
        
        # stratified fold generator
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)
        kf.get_n_splits(features)
    
        # If training an MTL model
        if model.startswith('MTL'):
            # Since y_MTL has 5 labels, we stratify the folds WRT the overall label
            labels_for_stratification = labels_dict.get('Overall')
            for train, val in kf.split(features, labels_for_stratification):
                t = (features[train], [e[train] for e in labels])
                v = (features[val], [e[val] for e in labels])
                output.append((t,v))
        # If training an overall model
        else:
            for train, val in kf.split(features, labels):
                t = (features[train], labels[train])
                v = (features[val], labels[val])
                output.append((t,v))
     
    # Else, if performing grid search, provide the training and internal partitions as they are
    elif (stage=="GS"):
        y_train_MTL = [y_train_MTL.iloc[:,0].values, 
                       y_train_MTL.iloc[:,1].values, 
                       y_train_MTL.iloc[:,2].values, 
                       y_train_MTL.iloc[:,3].values, 
                       y_train_MTL.iloc[:,4].values]
        
        y_internal_MTL = [y_internal_MTL.iloc[:,0].values, 
                       y_internal_MTL.iloc[:,1].values, 
                       y_internal_MTL.iloc[:,2].values, 
                       y_internal_MTL.iloc[:,3].values, 
                       y_internal_MTL.iloc[:,4].values]

        
        labels_dict = {
            'MTL':(y_train_MTL, y_internal_MTL),
            'Overall':(y_train_overall.values , y_internal_overall.values)
        }
        
        y = labels_dict.get(model)
        
        t = (X_train, y[0])
        v = (X_internal, y[1])
        # In this case, the output is a tuple containing the train and internal partitions
        output =(t,v)
        
    # Else, if evaluating on the external validation, just provide the external partition
    elif (stage=="EVAL"):
        y_external_MTL = [y_external_MTL.iloc[:,0].values, 
                       y_external_MTL.iloc[:,1].values, 
                       y_external_MTL.iloc[:,2].values, 
                       y_external_MTL.iloc[:,3].values, 
                       y_external_MTL.iloc[:,4].values]
        
        labels_dict = {
            'MTL':y_external_MTL,
            'Overall':y_external_overall.values
        }
        
        y = labels_dict.get(model)
        
        e = (X_external, y)
        # In this case, the output is the external partition
        output = e
    
    # Else, signal error and return empty tuple
    else:
        print("Error - Did not specify the stage in the experimental workflow. Returning empty tuple.")
        
    return output