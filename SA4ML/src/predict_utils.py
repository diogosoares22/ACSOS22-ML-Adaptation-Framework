import numpy as np
import pandas as pd

def compute_TPR(TP, FN):
    return TP / (TP + FN)

def compute_TNR(TN, FP):
    return TN / (TN + FP)

def compute_fraud_rate(TP, TN, FP, FN):
    return (TP + FN) / (TP + FP + TN + FN)

def get_entropy(probs): 
    return np.sum(np.multiply(probs, np.log(probs + 1e-20))  , axis=1)

def get_general_entropy(probs, predictions):

    entropy = get_entropy(probs)

    expected_predictions = np.argmax(probs, axis=-1)

    inverted_indices = (expected_predictions != predictions)

    entropy[inverted_indices] = entropy[inverted_indices] * (-1) - 2

    return entropy 

def get_max_conf(probs):
    return np.max(probs, axis=-1)

def find_ATC_threshold(scores, labels): 
    sorted_idx = np.argsort(scores)
    
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    fp = np.sum(labels==0)
    fn = 0.0
    
    min_fp_fn = np.abs(fp - fn)
    thres = 0.0
    for i in range(len(labels)): 
        if sorted_labels[i] == 0: 
            fp -= 1
        else: 
            fn += 1
        
        if np.abs(fp - fn) < min_fp_fn: 
            min_fp_fn = np.abs(fp - fn)
            thres = sorted_scores[i]
    
    return min_fp_fn, thres

def get_ATC_acc(thres, scores): 
    return np.mean(scores>=thres)*100.0

def get_ATC_tpr(mask, predictions):
    
    TP = np.sum(np.asarray((mask) & (predictions)))
    FN = np.sum(np.asarray((~mask) & (~predictions)))

    if (TP + FN) == 0:
        return 0
    
    return TP / (TP + FN)

def get_ATC_tnr(mask, predictions):
    TN = np.sum(np.asarray((mask) & (~predictions)))
    FP = np.sum(np.asarray((~mask) & (predictions)))
    
    if (TN + FP) == 0:
        return 0
    
    return TN / (TN + FP)

def get_ATC_fraud_rate(mask, predictions):
    TP = np.sum(np.asarray((mask) & (predictions)))
    TN = np.sum(np.asarray((mask) & (~predictions)))
    FN = np.sum(np.asarray((~mask) & (~predictions)))
    FP = np.sum(np.asarray((~mask) & (predictions)))
    
    return (TP + FN) / (TP + FP + TN + FN)

def get_correct_predictions_mask_with_classes(val_pred_probs, val_preds, val_labels, test_pred_probs, test_preds):
    no_classes = val_pred_probs.shape[1]
    
    val_scores = get_general_entropy(val_pred_probs, val_preds)

    test_scores = get_general_entropy(test_pred_probs, test_preds)

    mask = np.zeros(test_scores.shape)

    for i_class in range(no_classes):

        selected_indices = np.where(val_preds == i_class)
        
        class_scores = val_scores[selected_indices]

        class_labels = val_labels[selected_indices]

        class_preds = val_preds[selected_indices]

        _, ATC_threshold = find_ATC_threshold(class_scores, class_labels == class_preds)

        selected_test_indices = np.where(test_preds == i_class)

        class_test_scores = test_scores[selected_test_indices]

        mask[selected_test_indices] = np.asarray(class_test_scores >= ATC_threshold)

    return mask

def get_correct_predictions_mask_without_classes(val_pred_probs, val_preds, val_labels, test_pred_probs, test_preds):

    val_scores = get_general_entropy(val_pred_probs, val_preds)

    test_scores = get_general_entropy(test_pred_probs, test_preds)

    _, ATC_threshold = find_ATC_threshold(val_scores, val_labels == val_preds)

    mask = np.asarray(test_scores >= ATC_threshold)

    return mask

def predict_confusion_matrix(val_pred_probs, val_predictions, val_labels, test_pred_probs, test_predictions, classes=True):

    if classes:
        mask = get_correct_predictions_mask_with_classes(val_pred_probs, val_predictions, val_labels, test_pred_probs, test_predictions)
    else:
        mask = get_correct_predictions_mask_without_classes(val_pred_probs, val_predictions, val_labels, test_pred_probs, test_predictions)

    mask = mask.astype(bool)
    test_predictions = test_predictions.astype(bool)

    tpr = get_ATC_tpr(mask, test_predictions)
    tnr = get_ATC_tnr(mask, test_predictions)

    fraud_rate = get_ATC_fraud_rate(mask, test_predictions)

    return tpr, tnr, fraud_rate

def estimate_confusion_matrix(method: str, retrain_delta: int, metrics: dict):
    
    delay = int(method.split("_")[-1])

    # metrics of the current time interval
    test_scores = metrics["scores"][-1]
    test_predictions = metrics["predictions"][-1]
    test_scores = np.stack((1 - test_scores, test_scores)).T

    # Only 1 chunk of data ==> small validation set
    if "small" in method:
        # metrics of time interval "current time instant - delay"
        val_labels = metrics["real_labels"][-(1 + delay)]
        val_scores = metrics["scores"][-(1 + delay)]
        val_predictions = metrics["predictions"][-(1 + delay)]
        val_scores = np.stack((1 - val_scores, val_scores)).T
    
    # All chunks of data between delay_prev and delay_curr ==> big validation set
    else:
        # metrics from  "last retrain time - delay" to "current time instant - delay"
        print(f"len(metrics)={len(metrics['real_labels'])}   accessing index {-(1 + delay + retrain_delta)} to {-(1 + delay)}")
        all_val_labels = metrics["real_labels"][-(1 + delay + retrain_delta):-(1 + delay)]
        all_val_scores = metrics["scores"][-(1 + delay + retrain_delta):-(1 + delay)]
        all_val_predictions = metrics["predictions"][-(1 + delay + retrain_delta):-(1 + delay)]

        print(all_val_labels)

        val_labels = np.concatenate(all_val_labels)
        print(val_labels)
        val_scores = np.concatenate(all_val_scores)
        val_predictions = np.concatenate(all_val_predictions)
        val_scores = np.stack((1 - val_scores, val_scores)).T

    classes = False
    if "cbatc" in method:
        classes = True
        
    tpr, tnr, fraud_rate = predict_confusion_matrix(val_scores, val_predictions, val_labels, test_scores, test_predictions, classes=classes)

    return tpr, tnr, (1-tpr), (1-tnr), fraud_rate