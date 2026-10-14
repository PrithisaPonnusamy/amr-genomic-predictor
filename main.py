"""
Bacterial Antimicrobial Resistance (AMR) Prediction Project
===========================================================
End-to-end execution script:
1. Generates simulated continuous genomic features (3000 isolates, 95% susceptible, 5% resistant)
   across mock geographic regions (Region A, Region B, Region C).
2. Evaluates the unweighted Baseline Random Forest Classifier on a held-out test split.
3. Upgrades to an imbalanced-learn Pipeline with SMOTE-Tomek hybrid resampling under
   a strict 5-fold StratifiedKFold CV strategy to eliminate validation leakage.
4. Executes the Outbreak-Shielded Cross-Validation strategy (Training on Regions A & B,
   evaluating exclusively on unseen Region C) to evaluate geographic generalization.
"""

import os
import pandas as pd
from src.simulate_genomic_data import simulate_amr_genomic_dataset
from src.train_amr_model import run_amr_baseline_pipeline
from src.train_smotetomek_pipeline import run_stratified_smotetomek_cv
from src.train_outbreak_shielded import run_outbreak_shielded_pipeline
from src.clinical_card_reporter import generate_clinical_card_report


def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "synthetic_amr_genomic_data.csv")

    print("[Step 1/5] Simulating realistic bacterial genomic dataset with geographic regions...")
    df = simulate_amr_genomic_dataset(n_samples=3000, resistant_prevalence=0.05, random_seed=42)
    df.to_csv(data_path, index=False)
    print(f"Dataset saved to '{data_path}' with {len(df)} isolates.\n")

    print("[Step 2/5] Inspecting DataFrame Head & Verifying 95/5 Skew...")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("DataFrame Head:\n", df.head())
    print("\nTarget Label Distribution (amr_phenotype):")
    counts = df["amr_phenotype"].value_counts()
    percents = df["amr_phenotype"].value_counts(normalize=True) * 100
    dist_df = pd.DataFrame({"Count": counts, "Percentage (%)": percents})
    dist_df.index = ["Susceptible (0)", "Resistant Superbug (1)"]
    print(dist_df, "\n")

    print("[Step 3/5] Evaluating Baseline Random Forest Classifier (Stratified Test Split)...")
    run_amr_baseline_pipeline(data_path=data_path, test_size=0.20, random_seed=42)

    print("\n[Step 4/5] Executing Upgraded imbalanced-learn Pipeline with SMOTE-Tomek (5-Fold StratifiedKFold)...")
    run_stratified_smotetomek_cv(data_path=data_path, n_splits=5, random_seed=42)

    print("\n[Step 5/5] Executing Outbreak-Shielded Cross-Validation (Regions A & B -> Unseen Region C)...")
    run_outbreak_shielded_pipeline(data_path=data_path, random_seed=42)


if __name__ == "__main__":
    main()
