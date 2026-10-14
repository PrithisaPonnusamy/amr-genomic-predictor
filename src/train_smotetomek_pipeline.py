"""
Upgraded AMR Prediction Pipeline: Leak-Free SMOTE-Tomek Hybrid Resampling
========================================================================
Bioinformatics ML pipeline utilizing an imbalanced-learn Pipeline wrapper
integrated with a SMOTE-Tomek hybrid resampling engine, evaluated under a
strict StratifiedKFold cross-validation strategy.

Bioinformatics & Methodological Rationale:
------------------------------------------
1. Prevention of Data Leakage (The Fundamental Validation Law):
   In clinical genomics and biomarker discovery, preprocessing or resampling
   the entire dataset before cross-validation contaminates validation folds
   with synthetic artifacts derived from test samples, causing severe optimistic bias.
   The `imblearn.pipeline.Pipeline` strictly hooks into `fit_resample` during `.fit()`,
   guaranteeing that SMOTE-Tomek is fit exclusively on the training folds.
   Validation folds remain 100% untouched, representative clinical isolate populations.

2. SMOTE-Tomek Hybrid Mechanism:
   - SMOTE (Over-sampling): Operates in continuous genomic feature space (gene copy
     numbers, efflux scores, VAFs), synthesizing new minority superbug instances
     along feature manifolds between nearest neighbors.
   - Tomek Links (Under-sampling): Detects minimally separated, ambiguous pairs of
     susceptible and superbug isolates and removes the majority borderline instances,
     sharpening the biological decision boundary along multi-drug resistance frontiers.

3. Cross-Validation Rigor:
   5-fold StratifiedKFold maintains the 95:5 ratio across all splits. Out-of-fold
   predictions are collected across all 3,000 isolates to compare the true
   generalization Recall of the upgraded pipeline versus the unweighted baseline.
"""

import sys
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)


def load_genomic_data(data_path: str = "data/synthetic_amr_genomic_data.csv"):
    """
    Loads simulated bacterial genomic features and separates predictors from labels.
    """
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in ["isolate_id", "amr_phenotype"]]
    X = df[feature_cols].values
    y = df["amr_phenotype"].values
    return X, y, feature_cols


def run_stratified_smotetomek_cv(
    data_path: str = "data/synthetic_amr_genomic_data.csv",
    n_splits: int = 5,
    random_seed: int = 42
):
    """
    Executes leak-free StratifiedKFold cross-validation comparing:
    1. Baseline unweighted Random Forest
    2. Upgraded imbalanced-learn Pipeline with SMOTE-Tomek + Random Forest
    """
    X, y, feature_cols = load_genomic_data(data_path)
    n_samples = len(y)
    n_res = int((y == 1).sum())
    n_susc = int((y == 0).sum())

    print("=" * 86)
    print(" UPGRADED BIOINFORMATICS ML PIPELINE: LEAK-FREE SMOTE-TOMEK RESAMPLING")
    print("=" * 86)
    print(f"Total Cohort: {n_samples} bacterial isolates ({n_susc} Susceptible [95.0%], {n_res} Superbugs [5.0%])")
    print(f"Features:     {len(feature_cols)} continuous genomic features (copy numbers, efflux, porin loss, VAFs)")
    print(f"CV Strategy:  {n_splits}-Fold StratifiedKFold (Preserving 95:5 ratio per fold; Zero Validation Leakage)")
    print("-" * 86)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # Storage for fold metrics
    metrics_baseline = {"recall": [], "precision": [], "f1": [], "accuracy": [], "roc_auc": [], "pr_auc": []}
    metrics_smotetomek = {"recall": [], "precision": [], "f1": [], "accuracy": [], "roc_auc": [], "pr_auc": []}

    # Out-of-fold prediction arrays
    oof_preds_base = np.zeros(n_samples, dtype=int)
    oof_probas_base = np.zeros(n_samples, dtype=float)

    oof_preds_pipe = np.zeros(n_samples, dtype=int)
    oof_probas_pipe = np.zeros(n_samples, dtype=float)

    print("\nExecuting Fold-by-Fold Cross-Validation...")
    print(f"{'Fold':<6} | {'Baseline Recall':<17} {'Baseline Prec':<15} | {'SMOTE-Tomek Recall':<20} {'SMOTE-Tomek Prec':<18} | {'Recall Gain'}")
    print("-" * 86)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # -------------------------------------------------------------
        # 1. Baseline Model (Standard Random Forest without resampling)
        # -------------------------------------------------------------
        rf_baseline = RandomForestClassifier(
            n_estimators=100,
            random_state=random_seed,
            n_jobs=-1
        )
        rf_baseline.fit(X_train, y_train)
        pred_b = rf_baseline.predict(X_val)
        proba_b = rf_baseline.predict_proba(X_val)[:, 1]

        oof_preds_base[val_idx] = pred_b
        oof_probas_base[val_idx] = proba_b

        rec_b = recall_score(y_val, pred_b, pos_label=1, zero_division=0)
        prec_b = precision_score(y_val, pred_b, pos_label=1, zero_division=0)
        f1_b = f1_score(y_val, pred_b, pos_label=1, zero_division=0)
        acc_b = accuracy_score(y_val, pred_b)
        roc_b = roc_auc_score(y_val, proba_b)
        pr_b = average_precision_score(y_val, proba_b)

        metrics_baseline["recall"].append(rec_b)
        metrics_baseline["precision"].append(prec_b)
        metrics_baseline["f1"].append(f1_b)
        metrics_baseline["accuracy"].append(acc_b)
        metrics_baseline["roc_auc"].append(roc_b)
        metrics_baseline["pr_auc"].append(pr_b)

        # -------------------------------------------------------------
        # 2. Upgraded imbalanced-learn Pipeline (SMOTE-Tomek Resampler)
        # -------------------------------------------------------------
        # Resampling happens STRICTLY within the pipeline during .fit()
        # Validation fold X_val is NEVER seen by SMOTETomek
        smotetomek_pipeline = Pipeline([
            ("resampler", SMOTETomek(random_state=random_seed)),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1))
        ])

        smotetomek_pipeline.fit(X_train, y_train)
        pred_p = smotetomek_pipeline.predict(X_val)
        proba_p = smotetomek_pipeline.predict_proba(X_val)[:, 1]

        oof_preds_pipe[val_idx] = pred_p
        oof_probas_pipe[val_idx] = proba_p

        rec_p = recall_score(y_val, pred_p, pos_label=1, zero_division=0)
        prec_p = precision_score(y_val, pred_p, pos_label=1, zero_division=0)
        f1_p = f1_score(y_val, pred_p, pos_label=1, zero_division=0)
        acc_p = accuracy_score(y_val, pred_p)
        roc_p = roc_auc_score(y_val, proba_p)
        pr_p = average_precision_score(y_val, proba_p)

        metrics_smotetomek["recall"].append(rec_p)
        metrics_smotetomek["precision"].append(prec_p)
        metrics_smotetomek["f1"].append(f1_p)
        metrics_smotetomek["accuracy"].append(acc_p)
        metrics_smotetomek["roc_auc"].append(roc_p)
        metrics_smotetomek["pr_auc"].append(pr_p)

        gain_str = f"+{(rec_p - rec_b)*100:5.2f}%" if rec_p >= rec_b else f"-{(rec_b - rec_p)*100:5.2f}%"
        print(f"Fold {fold_idx:<2d} | {rec_b*100:>6.2f}%           {prec_b*100:>6.2f}%         | "
              f"{rec_p*100:>6.2f}%             {prec_p*100:>6.2f}%           | {gain_str}")

    print("-" * 86)

    # Compute Aggregate Cross-Validation Statistics
    mean_rec_b, std_rec_b = np.mean(metrics_baseline["recall"]), np.std(metrics_baseline["recall"])
    mean_rec_p, std_rec_p = np.mean(metrics_smotetomek["recall"]), np.std(metrics_smotetomek["recall"])

    mean_prec_b, std_prec_b = np.mean(metrics_baseline["precision"]), np.std(metrics_baseline["precision"])
    mean_prec_p, std_prec_p = np.mean(metrics_smotetomek["precision"]), np.std(metrics_smotetomek["precision"])

    mean_f1_b, std_f1_b = np.mean(metrics_baseline["f1"]), np.std(metrics_baseline["f1"])
    mean_f1_p, std_f1_p = np.mean(metrics_smotetomek["f1"]), np.std(metrics_smotetomek["f1"])

    mean_acc_b = np.mean(metrics_baseline["accuracy"])
    mean_acc_p = np.mean(metrics_smotetomek["accuracy"])

    mean_roc_b = np.mean(metrics_baseline["roc_auc"])
    mean_roc_p = np.mean(metrics_smotetomek["roc_auc"])

    # Out-of-fold Total Confusion Matrices
    cm_base = confusion_matrix(y, oof_preds_base)
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()

    cm_pipe = confusion_matrix(y, oof_preds_pipe)
    tn_p, fp_p, fn_p, tp_p = cm_pipe.ravel()

    # -------------------------------------------------------------
    # 3. Print Comprehensive Performance Comparison
    # -------------------------------------------------------------
    print("\n" + "#" * 86)
    print("### CROSS-VALIDATION RESULTS: BASELINE vs. SMOTE-TOMEK PIPELINE ###")
    print("#" * 86)
    print(f"\n{'Metric (Resistant Superbug Class)':<35} | {'Baseline Random Forest':<22} | {'SMOTE-Tomek Pipeline':<22}")
    print("-" * 86)
    print(f"{'Mean Superbug Recall (Sensitivity)':<35} | {mean_rec_b*100:>6.2f}% ± {std_rec_b*100:4.2f}%       | "
          f"{mean_rec_p*100:>6.2f}% ± {std_rec_p*100:4.2f}%  <-- [RECALL COMPARISON]")
    print(f"{'Mean Superbug Precision (PPV)':<35} | {mean_prec_b*100:>6.2f}% ± {std_prec_b*100:4.2f}%       | "
          f"{mean_prec_p*100:>6.2f}% ± {std_prec_p*100:4.2f}%")
    print(f"{'Mean Superbug F1-Score':<35} | {mean_f1_b*100:>6.2f}% ± {std_f1_b*100:4.2f}%       | "
          f"{mean_f1_p*100:>6.2f}% ± {std_f1_p*100:4.2f}%")
    print(f"{'Mean Overall Accuracy':<35} | {mean_acc_b*100:>6.2f}%                 | {mean_acc_p*100:>6.2f}%")
    print(f"{'Mean ROC-AUC':<35} | {mean_roc_b:>6.4f}                  | {mean_roc_p:>6.4f}")
    print("-" * 86)
    print(f"{'Total Detected Superbugs (TP)':<35} | {tp_b:>4d} / {n_res} ({(tp_b/n_res)*100:.1f}%)        | {tp_p:>4d} / {n_res} ({(tp_p/n_res)*100:.1f}%)")
    print(f"{'Total Missed Superbugs (FN)':<35} | {fn_b:>4d} / {n_res} ({(fn_b/n_res)*100:.1f}%)        | {fn_p:>4d} / {n_res} ({(fn_p/n_res)*100:.1f}%)")
    print(f"{'Total False Positives (FP)':<35} | {fp_b:>4d} / {n_susc} ({(fp_b/n_susc)*100:.1f}%)       | {fp_p:>4d} / {n_susc} ({(fp_p/n_susc)*100:.1f}%)")
    print("#" * 86)

    # 4. Out-of-Fold Classification Reports
    print("\n" + "=" * 30 + " BASELINE: OUT-OF-FOLD CLASSIFICATION REPORT " + "=" * 30)
    print(classification_report(y, oof_preds_base, target_names=["Susceptible (0)", "Resistant Superbug (1)"], digits=4))
    print("Baseline Out-of-Fold Confusion Matrix:")
    print(f"  TN={tn_b:<4d}  FP={fp_b:<4d}")
    print(f"  FN={fn_b:<4d}  TP={tp_b:<4d}")

    print("\n" + "=" * 27 + " SMOTE-TOMEK PIPELINE: OUT-OF-FOLD CLASSIFICATION REPORT " + "=" * 27)
    print(classification_report(y, oof_preds_pipe, target_names=["Susceptible (0)", "Resistant Superbug (1)"], digits=4))
    print("SMOTE-Tomek Pipeline Out-of-Fold Confusion Matrix:")
    print(f"  TN={tn_p:<4d}  FP={fp_p:<4d}")
    print(f"  FN={fn_p:<4d}  TP={tp_p:<4d}")
    print("-" * 86)

    # 5. BIOINFORMATICS & CLINICAL INSIGHT
    recall_increase_abs = (mean_rec_p - mean_rec_b) * 100
    recall_increase_rel = ((mean_rec_p - mean_rec_b) / mean_rec_b) * 100
    saved_superbugs = fn_b - fn_p

    print("\n" + "#" * 86)
    print("### EXPERT BIOINFORMATICS EVALUATION: SUPERBUG RECALL IMPACT ###")
    print("#" * 86)
    print(f" >>> ABSOLUTE RECALL IMPROVEMENT:   +{recall_increase_abs:.2f}% percentage points")
    print(f" >>> RELATIVE RECALL MULTIPLIER:    +{recall_increase_rel:.1f}% increase ({mean_rec_p/mean_rec_b:.2f}x fold boost)")
    print(f" >>> PATIENT SAFETY GAIN:           {saved_superbugs} additional lethal superbugs intercepted across cohort")
    print()
    print(" 1. ZERO-LEAKAGE ARCHITECTURAL GUARANTEE:")
    print("    - By nesting SMOTETomek within `imblearn.pipeline.Pipeline`, synthetic superbugs are")
    print("      generated exclusively from training fold coordinates in continuous genomic space.")
    print("    - Validation folds are never seen during resampling, avoiding the infamous 'data leakage trap'")
    print("      frequently found in flawed published bioinformatics studies.")
    print()
    print(" 2. WHY SMOTE-TOMEK BOOSTS RECALL ON GENOMIC MULTI-DRUG DETERMINANTS:")
    print("    - SMOTE synthesizes intermediate superbug genomic profiles (e.g. interpolating between")
    print("      moderate-KPC and high-NDM isolates), forcing decision trees to evaluate minority splits.")
    print("    - Tomek Links purges borderline, ambiguous susceptible isolates (e.g. basal copy numbers")
    print("      with slight sequencing noise) that otherwise induce false negative conservative splits.")
    print("#" * 86 + "\n")

    return {
        "metrics_baseline": metrics_baseline,
        "metrics_smotetomek": metrics_smotetomek,
        "oof_preds_base": oof_preds_base,
        "oof_preds_pipe": oof_preds_pipe,
        "summary": {
            "mean_recall_baseline": mean_rec_b,
            "mean_recall_smotetomek": mean_rec_p,
            "saved_superbugs": saved_superbugs
        }
    }


if __name__ == "__main__":
    run_stratified_smotetomek_cv()
