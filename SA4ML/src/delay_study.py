#!/usr/bin/env python

from predict_utils import *
import defs
import pandas as pd

def main():

    print(
        "[D] --------------------------- GENERATE DELAY RESULTS ---------------------------"
    )
    

    DATASET_PATH = defs.BASE_DATASETS_PATH + "pre-generated/new/timeInterval_10-rand_sample.pkl"
    DATASET_SAVE_PATH = defs.BASE_DATASETS_PATH + "pre-generated/new/timeInterval_10-rand_sample_after_delay_metrics.pkl"
    METRICS_PATH = defs.BASE_DATASETS_PATH + "pre-generated/tmp/"
    TIME_INTERVAL = 10
    MAX_DELAY = 15
    DELAY_PERIODS = range(1, MAX_DELAY)

    print(
        f"\n[D] Load Results"
    )

    dataset = pd.read_pickle(DATASET_PATH)
    metrics = [pd.read_pickle(METRICS_PATH + "metrics-timeInterval_10-noRetrain-seed_1.pkl")]
    metrics += [pd.read_pickle(METRICS_PATH + f"metrics-timeInterval_10-retrainPeriodHours_{i}-retrainMode_single-seed_1.pkl") for i in range(10, defs.MAX_TIME, 10)]

    print(
        f"\n[D] Add Confusion Matrix Rates"
    )

    # transform confusion matrix features into rates
    for prefix in ["curr", "prev", "next2prev"]:
        dataset[f"{prefix}-TPR"] = dataset[f"{prefix}_tp"] / (
            dataset[f"{prefix}_tp"] + dataset[f"{prefix}_fn"]
        )
        dataset[f"{prefix}-TNR"] = dataset[f"{prefix}_tn"] / (
            dataset[f"{prefix}_tn"] + dataset[f"{prefix}_fp"]
        )
        dataset[f"{prefix}-FNR"] = dataset[f"{prefix}_fn"] / (
            dataset[f"{prefix}_tp"] + dataset[f"{prefix}_fn"]
        )
        dataset[f"{prefix}-FPR"] = dataset[f"{prefix}_fp"] / (
            dataset[f"{prefix}_tn"] + dataset[f"{prefix}_fp"]
        )
    
    print(
        f"\n[D] Add Deltas"
    )

    # no delay
    dataset["delta-TPR-0"] = dataset["avg-TPR-retrain"] - dataset["curr-TPR"]
    dataset["delta-TNR-0"] = dataset["avg-TNR-retrain"] - dataset["curr-TNR"]
    dataset["delta-TPR-nop-0"] = dataset["avg-TPR-no_retrain"] - dataset["curr-TPR"]
    dataset["delta-TNR-nop-0"] = dataset["avg-TNR-no_retrain"] - dataset["curr-TNR"]

    # add deltas
    for delay in DELAY_PERIODS:
        for index , row in dataset.iterrows():
            prev_hour, curr_hour = row["prev_retrain_hour"], row["curr_retrain_hour"]
            delay_prev_hour, delay_curr_hour = prev_hour - delay * TIME_INTERVAL, curr_hour - delay * TIME_INTERVAL

            if delay_prev_hour < 0 or (curr_hour + TIME_INTERVAL) >= defs.MAX_TIME:
                continue

            desired_metrics_before_training = metrics[delay_prev_hour // TIME_INTERVAL]
            desired_metrics_after_training = metrics[delay_curr_hour // TIME_INTERVAL]

            res_before_training = desired_metrics_before_training.loc[curr_hour // TIME_INTERVAL]
            res_after_no_training = desired_metrics_before_training.loc[(curr_hour // TIME_INTERVAL) + 1]
            res_after_training = desired_metrics_after_training.loc[(curr_hour // TIME_INTERVAL) + 1]
            
            dataset.at[index,'delta-TPR-{}'.format(delay)] = compute_TPR(res_after_training["count_tp"], res_after_training["count_fn"]) - compute_TPR(res_before_training["count_tp"], res_before_training["count_fn"])
            dataset.at[index,'delta-TNR-{}'.format(delay)] = compute_TNR(res_after_training["count_tn"], res_after_training["count_fp"]) - compute_TNR(res_before_training["count_tn"], res_before_training["count_fp"])
            dataset.at[index,'delta-TPR-nop-{}'.format(delay)] = compute_TPR(res_after_no_training["count_tp"], res_after_no_training["count_fn"]) - compute_TPR(res_before_training["count_tp"], res_before_training["count_fn"])
            dataset.at[index,'delta-TNR-nop-{}'.format(delay)] = compute_TNR(res_after_no_training["count_tn"], res_after_no_training["count_fp"]) - compute_TNR(res_before_training["count_tn"], res_before_training["count_fp"])

        print(
            f"\n        [L] Delay {delay}"
        )

    print(
        f"\n[D] Add Delay Metrics"
    )

    # add delay metrics
    for delay in DELAY_PERIODS:
        for index , row in dataset.iterrows():
            prev_hour, curr_hour = row["prev_retrain_hour"], row["curr_retrain_hour"]
            delay_prev_hour, delay_curr_hour = prev_hour - delay * TIME_INTERVAL, curr_hour - delay * TIME_INTERVAL

            if delay_prev_hour < 0:
                continue

            desired_metrics_before_training = metrics[delay_prev_hour // TIME_INTERVAL]
            
            delay_curr_before_training = desired_metrics_before_training.loc[delay_curr_hour // TIME_INTERVAL]
            curr_before_training = desired_metrics_before_training.loc[curr_hour // TIME_INTERVAL]
            
            dataset.at[index,'delayed-TPR-{}'.format(delay)] = compute_TPR(delay_curr_before_training["count_tp"], delay_curr_before_training["count_fn"])
            dataset.at[index,'delayed-TNR-{}'.format(delay)] = compute_TNR(delay_curr_before_training["count_tn"], delay_curr_before_training["count_fp"])
            dataset.at[index,'delayed_fraud_rate-{}'.format(delay)] = compute_fraud_rate(delay_curr_before_training["count_tp"], delay_curr_before_training["count_tn"]
                                                                                        , delay_curr_before_training["count_fp"], delay_curr_before_training["count_fn"])

            dataset.at[index,'curr-TPR-{}'.format(delay)] = compute_TPR(curr_before_training["count_tp"], curr_before_training["count_fn"])
            dataset.at[index,'curr-TNR-{}'.format(delay)] = compute_TNR(curr_before_training["count_tn"], curr_before_training["count_fp"])
            dataset.at[index,'curr_fraud_rate-{}'.format(delay)] = compute_fraud_rate(curr_before_training["count_tp"], curr_before_training["count_tn"]
                                                                                        , curr_before_training["count_fp"], curr_before_training["count_fn"])
        
        print(
            f"\n        [L] Delay {delay}"
        )

    print(
        f"\n[D] Add Delay Predictions"
    )

    # add predictions
    for delay in DELAY_PERIODS:
        for index , row in dataset.iterrows():
            prev_hour, curr_hour = row["prev_retrain_hour"], row["curr_retrain_hour"]
            delay_prev_hour, delay_curr_hour = prev_hour - delay * TIME_INTERVAL, curr_hour - delay * TIME_INTERVAL

            if delay_prev_hour < 0:
                continue
            
            desired_metrics_before_training = metrics[delay_prev_hour // TIME_INTERVAL]
            
            validation = desired_metrics_before_training.loc[delay_curr_hour // TIME_INTERVAL]
            test = desired_metrics_before_training.loc[curr_hour // TIME_INTERVAL]

            # Only 1 chunk of data

            val_labels = validation["real_labels"]
            val_scores = validation["scores"]
            val_predictions = validation["predictions"]
            val_scores = np.stack((1 - val_scores, val_scores)).T

            test_scores = test["scores"]
            test_predictions = test["predictions"]
            test_scores = np.stack((1 - test_scores, test_scores)).T

            TPR, TNR, fraud_rate = predict_confusion_matrix(val_scores, val_predictions, val_labels, test_scores, test_predictions, classes=True)
            
            dataset.at[index,'predict-TPR-{}-with-classes'.format(delay)] = TPR
            dataset.at[index,'predict-TNR-{}-with-classes'.format(delay)] = TNR
            dataset.at[index,'predict_fraud_rate-{}-with-classes'.format(delay)] = fraud_rate

            TPR, TNR, fraud_rate = predict_confusion_matrix(val_scores, val_predictions, val_labels, test_scores, test_predictions, classes=False)

            dataset.at[index,'predict-TPR-{}-without-classes'.format(delay)] = TPR
            dataset.at[index,'predict-TNR-{}-without-classes'.format(delay)] = TNR
            dataset.at[index,'predict_fraud_rate-{}-without-classes'.format(delay)] = fraud_rate

            # All chunks of data between delay_prev and delay_curr

            all_val_labels = []
            all_val_scores = []
            all_val_predictions = [] 

            for t in range(delay_prev_hour + TIME_INTERVAL, delay_curr_hour + TIME_INTERVAL, TIME_INTERVAL):
                validation = desired_metrics_before_training.loc[t // TIME_INTERVAL]
                all_val_labels += validation["real_labels"]
                all_val_scores += validation["scores"]
                all_val_predictions += validation["predictions"]

            val_labels = np.concatenate(all_val_labels)
            val_scores = np.concatenate(all_val_scores)
            val_predictions = np.concatenate(all_val_predictions)
            val_scores = np.stack((1 - val_scores, val_scores)).T

            TPR, TNR, fraud_rate = predict_confusion_matrix(val_scores, val_predictions, val_labels, test_scores, test_predictions, classes=True)
            
            dataset.at[index,'predict-TPR-{}-with-classes-with-all-data'.format(delay)] = TPR
            dataset.at[index,'predict-TNR-{}-with-classes-with-all-data'.format(delay)] = TNR
            dataset.at[index,'predict_fraud_rate-{}-with-classes-with-all-data'.format(delay)] = fraud_rate

            TPR, TNR, fraud_rate = predict_confusion_matrix(val_scores, val_predictions, val_labels, test_scores, test_predictions, classes=False)

            dataset.at[index,'predict-TPR-{}-without-classes-with-all-data'.format(delay)] = TPR
            dataset.at[index,'predict-TNR-{}-without-classes-with-all-data'.format(delay)] = TNR
            dataset.at[index,'predict_fraud_rate-{}-without-classes-with-all-data'.format(delay)] = fraud_rate

        print(
            f"\n        [L] Delay {delay}"
        )

    print(
        f"\n[D] Save new data"
        )


    dataset.to_pickle(DATASET_SAVE_PATH) 


if __name__ == "__main__":
    main()
