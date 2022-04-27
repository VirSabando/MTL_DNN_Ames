import numpy as np 

# Based on the reults in terms of true positives (TP),
# True Negatives (TN), False Positives (FP) and
# False Negatives (FN), we compute all metrics reported in the paper.
def get_results(tp, tn, fp, fn):
    specificity = 0
    sensitivity = 0
    precision = 0
    accuracy = 0
    f1_score = 0
    h1_score = 0

    num_instances = tp + tn + fp + fn
    print('num of instances = '+ str(num_instances))

    if ((tn + fp) > 0):
        specificity = tn / (tn + fp)
    if ((tp + fp) > 0):
        precision = tp / (tp + fp)
    if ((tp + fn) > 0):
        sensitivity = tp / (tp + fn)
    if (num_instances > 0):
        accuracy = (tp + tn) / num_instances
    if ((2 * tp + fp + fn) > 0):
        f1_score = 2 * tp / (2 * tp + fp + fn)
    if ((specificity + sensitivity) > 0):
        h1_score = 2 * (specificity * sensitivity) / (specificity + sensitivity)
    balanced_accuracy = (specificity + sensitivity)/2

    tuple_results = (tp, tn, fp, fn)
    tuple_metrics = (specificity, sensitivity, precision, accuracy, balanced_accuracy, f1_score, h1_score)
    return (tuple_results, tuple_metrics)

# Compute metrics using get_results() based on the ground-truth labels (y_true) and the predictions of the DNN (y_pred)
def get_metrics(y_true,y_pred):
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    #auc_v = metrics.roc_auc_score(y_real, y_prob) #capaz que hay alguna mejor forma de calcularlo
    results_metrics = get_results(tp, tn, fp, fn)
    return results_metrics

# Helper function that allows to filter out those predictions having an undefined label (-1),
# which should NOT be taken into account to compute the metrics
# receives the ground-truth labels (true), the predicted label (pred) and the logits of the DNN (logit)
def filter_nan_A(true, pred, logit):
    idx = np.where(true != -1)
    return idx[0], true[idx[0]].reshape(-1,), pred[idx[0]].reshape(-1,), logit[idx[0]].reshape(-1,)

