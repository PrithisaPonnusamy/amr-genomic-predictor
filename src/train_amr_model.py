"""
Bacterial Antimicrobial Resistance (AMR) Prediction Pipeline
============================================================
Trains a baseline Random Forest Classifier to predict antimicrobial resistance
from continuous bacterial genomic features under severe class imbalance (95% Susceptible,
5% Resistant Superbugs).

Clinical Context:
-----------------
In clinical microbiology and infectious disease diagnostics, the primary safety metric is
RECALL (Sensitivity) for the resistant superbug class.
A FALSE NEGATIVE (classifying a true resistant superbug as susceptible) leads to fatal clinical
consequences: prescribing an inactive antibiotic (e.g., standard cephalosporins or carbapenems),
causing rapid clinical deterioration, septic shock, patient mortality, and nosocomial outbreak
dissemination in intensive care units (ICUs).
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def load_genomic_data(data_path: str = "data/synthetic_amr_genomic_data.csv"):
    """
    Loads simulated bacterial genomic features and separates predictors from labels.
    """
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in ["isolate_id", "amr_phenotype"]]
    X = df[feature_cols]
    y = df["amr_phenotype"]
    return X, y, feature_cols


def run_amr_baseline_pipeline(
    data_path: str = "data/synthetic_amr_genomic_data.csv",
    test_size: float = 0.20,
    random_seed: int = 42
):
    """
    Splits data, trains a baseline Random Forest classifier, prints the classification report,
    and emphasizes the Recall score for the resistant class.
    """
    X, y, feature_cols = load_genomic_data(data_path)

    # 1. Stratified Train-Test Split (crucial for maintaining 95:5 class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed
    )

    n_test = len(y_test)
    n_susc_test = int((y_test == 0).sum())
    n_res_test = int((y_test == 1).sum())

    print("=" * 82)
    print("  EXPERT BIOINFORMATICS ML PIPELINE: ANTIMICROBIAL RESISTANCE PREDICTION")
    print("=" * 82)
    print(f"Dataset Summary: {len(X)} bacterial isolates evaluated across {X.shape[1]} continuous genomic features.")
    print(f"Training Partition: {len(y_train)} isolates (Susceptible: {(y_train == 0).sum()}, Resistant: {(y_train == 1).sum()})")
    print(f"Test Partition:     {n_test} isolates (Susceptible: {n_susc_test} [{n_susc_test/n_test*100:.1f}%], "
          f"Resistant Superbugs: {n_res_test} [{n_res_test/n_test*100:.1f}%])")
    print("-" * 82)

    # 2. Train Baseline Random Forest Classifier
    # Note: Baseline model uses default uniform class weights and standard 0.5 decision threshold
    baseline_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_seed,
        n_jobs=-1
    )
    baseline_rf.fit(X_train, y_train)

    # 3. Model Inference on Unseen Test Isolates
    y_pred = baseline_rf.predict(X_test)
    y_proba = baseline_rf.predict_proba(X_test)[:, 1]

    # Metrics computation
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_str = classification_report(
        y_test,
        y_pred,
        target_names=["Susceptible (Class 0)", "Resistant Superbug (Class 1)"],
        digits=4
    )

    res_recall = recall_score(y_test, y_pred, pos_label=1)
    res_precision = precision_score(y_test, y_pred, pos_label=1)
    res_f1 = f1_score(y_test, y_pred, pos_label=1)
    accuracy = (tp + tn) / n_test
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    # 4. Print Full Classification Report
    print("\n" + "=" * 31 + " CLASSIFICATION REPORT " + "=" * 32)
    print(report_str)
    print("-" * 82)

    # 5. Print Confusion Matrix
    print("\n" + "=" * 33 + " CONFUSION MATRIX " + "=" * 33)
    print("                               Predicted Susceptible     Predicted Resistant Superbug")
    print(f"Actual Susceptible (Class 0):         {tn:<5d} (TN)                  {fp:<5d} (FP)")
    print(f"Actual Resistant   (Class 1):         {fn:<5d} (FN)                  {tp:<5d} (TP)")
    print("-" * 82)

    # 6. EMPHASIS: Resistant Superbug Class Recall
    print("\n" + "#" * 82)
    print("### CRITICAL METRIC EMPHASIS: RESISTANT SUPERBUG (CLASS 1) RECALL ###")
    print("#" * 82)
    print(f"\n  >> OVERALL ACCURACY:                     {accuracy * 100:.2f}%")
    print(f"  >> RESISTANT CLASS (1) RECALL:           {res_recall * 100:.2f}%  ({tp} / {n_res_test} superbugs detected)")
    print(f"  >> RESISTANT CLASS (1) PRECISION:        {res_precision * 100:.2f}%")
    print(f"  >> RESISTANT CLASS (1) F1-SCORE:         {res_f1 * 100:.2f}%")
    print(f"  >> FALSE NEGATIVES (MISSED SUPERBUGS):   {fn} out of {n_res_test} ({fn/n_res_test*100:.1f}%)")
    print(f"  >> FALSE POSITIVES:                      {fp} out of {n_susc_test} ({fp/n_susc_test*100:.1f}%)")
    print(f"  >> ROC-AUC:                              {roc_auc:.4f}")
    print(f"  >> PR-AUC (Average Precision):           {pr_auc:.4f}")

    print("\n" + "-" * 82)
    print("  CLINICAL & BIOINFORMATICS SIGNIFICANCE OF THE RECALL METRIC:")
    print("-" * 82)
    print(f"  1. THE 'ACCURACY PARADOX':")
    print(f"     While the baseline model boasts an impressive overall accuracy of {accuracy*100:.2f}%,")
    print(f"     it fails to detect {fn} out of {n_res_test} resistant superbugs (Recall: {res_recall*100:.2f}%).")
    print(f"     A trivial 'dumb' classifier that predicts every isolate as Susceptible would")
    print(f"     achieve 95.0% accuracy yet deliver 0% Recall, proving accuracy is clinically useless here.")
    print()
    print(f"  2. CLINICAL REPERCUSSIONS OF FALSE NEGATIVES (FN = {fn}):")
    print(f"     In clinical genomics, a False Negative is an existential medical error.")
    print(f"     - A patient with a carbapenemase/ESBL-producing superbug is misclassified as susceptible.")
    print(f"     - The clinician initiates standard empiric therapy (e.g., ceftriaxone, meropenem).")
    print(f"     - The pathogen continues unchecked, escalating into bacteremia, septic shock, or death.")
    print(f"     - Undetected superbugs evade hospital isolation protocols, driving nosocomial outbreaks.")
    print()
    print(f"  3. ASYMMETRIC LOSS FUNCTION (FALSE NEGATIVE vs. FALSE POSITIVE):")
    print(f"     - Cost of False Negative (FN): Patient mortality, therapeutic failure, transmission.")
    print(f"     - Cost of False Positive (FP): Administration of reserve, second-line antimicrobials")
    print(f"       (e.g., colistin, cefiderocol) and reflex confirmatory AST testing. While suboptimal,")
    print(f"       a false positive is vastly preferable to an untreated lethal infection.")
    print()
    print(f"  4. ROOT CAUSE IN MACHINE LEARNING ON IMBALANCED GENOMIC DATA:")
    print(f"     Standard Random Forest splits optimize Gini impurity across the global population.")
    print(f"     Because 95% of training samples are Susceptible, tree nodes prioritize purifying the")
    print(f"     majority class, setting implicit 0.5 probability boundaries that sacrifice minority recall.")
    print("#" * 82)

    # 7. Genomic Feature Importances
    print("\n" + "=" * 29 + " TOP GENOMIC PREDICTORS (GINI IMPORTANCE) " + "=" * 29)
    importances = baseline_rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_indices, start=1):
        feature_name = feature_cols[idx]
        imp_score = importances[idx]
        print(f"  {rank:2d}. {feature_name:<32} {imp_score*100:6.2f}% (Gini: {imp_score:.4f})")
    print("=" * 82)

    # 8. BIOINFORMATICS SOLUTION: Demonstrating Cost-Sensitive Threshold Tuning
    print("\n" + "=" * 28 + " CLINICAL MITIGATION: THRESHOLD ADJUSTMENT " + "=" * 28)
    print("  By calibrating the classification threshold for high sensitivity (Recall-oriented),")
    print("  we recover missed resistant superbugs from the baseline model's predicted probabilities:\n")
    
    thresholds = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10]
    print(f"  {'Threshold':<12} {'Resistant Recall':<18} {'Resistant Precision':<20} {'FN (Missed)':<14} {'FP'}")
    print("  " + "-" * 72)
    for thresh in thresholds:
        y_thresh_pred = (y_proba >= thresh).astype(int)
        rec = recall_score(y_test, y_thresh_pred, pos_label=1)
        prec = precision_score(y_test, y_thresh_pred, pos_label=1, zero_division=0)
        cm_t = confusion_matrix(y_test, y_thresh_pred)
        fn_t = cm_t[1, 0]
        fp_t = cm_t[0, 1]
        flag = " <-- (Baseline default)" if thresh == 0.50 else ""
        if rec >= 0.90 and thresh != 0.50 and flag == "":
            flag = " <-- (High Sensitivity Clinical Operating Point)"
        print(f"  {thresh:<12.2f} {rec*100:>6.2f}%           {prec*100:>6.2f}%              {fn_t:<14d} {fp_t:<5d}{flag}")
    print("=" * 82 + "\n")

    return {
        "model": baseline_rf,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": {
            "accuracy": accuracy,
            "res_recall": res_recall,
            "res_precision": res_precision,
            "res_f1": res_f1,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "tp": tp,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }
    }


if __name__ == "__main__":
    run_amr_baseline_pipeline()
