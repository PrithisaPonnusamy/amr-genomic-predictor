# Bacterial Antimicrobial Resistance (AMR) Genomic Prediction Pipeline

An expert bioinformatics machine learning project developed to predict high-level antimicrobial resistance (multidrug-resistant "superbugs") directly from bacterial continuous genomic features derived from whole-genome sequencing (WGS).

---

## 1. Scientific & Clinical Background

### The Clinical Stakes
In clinical microbiology and infectious disease surveillance, **Recall (Sensitivity)** for resistant superbugs is the primary safety metric. 
- **The Danger of a False Negative (FN):** If a carbapenemase- or ESBL-producing superbug is misclassified as susceptible, a clinician prescribes standard first-line beta-lactams. The bacterial infection continues unchecked, escalating into bacteremia, septic shock, ICU mortality, and uncontrolled nosocomial spread.
- **The Asymmetry with False Positives (FP):** A false positive causes escalation to second-line reserve agents (e.g., colistin, cefiderocol) or reflex phenotypic AST confirmatory testing. While costly, an FP is clinically survivable; a FN is frequently fatal.

### The 95:5 Class Imbalance & The "Accuracy Paradox"
In broad genomic screening and clinical isolate surveillance, wild-type or standard susceptible isolates represent the vast majority (~95%), while pan-resistant superbugs represent a high-consequence minority (~5%). 
A naive model predicting all isolates as "Susceptible" scores **95.0% accuracy** while having **0% Recall** on resistant superbugs.

---

## 2. Genomic Feature Architecture

The simulator models continuous-valued biological parameters measured via short- and long-read sequencing depth:

| Feature Name | Biological Description | Resistance Mechanism |
| :--- | :--- | :--- |
| `bla_KPC_copy_number` | KPC carbapenem-hydrolyzing class A beta-lactamase dosage | Enzymatic drug inactivation |
| `bla_NDM_copy_number` | NDM metallo-beta-lactamase (carbapenemase) dosage | Enzymatic drug inactivation |
| `bla_CTX_M_copy_number`| CTX-M extended-spectrum beta-lactamase (ESBL) dosage | Enzymatic drug inactivation |
| `bla_OXA_48_copy_number`| OXA-48-like class D oxacillinase dosage | Enzymatic drug inactivation |
| `acrAB_tolC_efflux_score` | RND multidrug efflux pump operon expression/dosage | Active drug extrusion |
| `ompK35_porin_loss_index` | Outer membrane porin K35 disruption index | Reduced membrane permeability |
| `ompK36_porin_loss_index` | Outer membrane porin K36 disruption index | Reduced membrane permeability |
| `gyrA_S83L_vaf` | DNA gyrase subunit A resistance mutation allele frequency | Target-site modification |
| `parC_S80I_vaf` | Topoisomerase IV subunit A mutation allele frequency | Target-site modification |
| `incF_plasmid_copy_ratio` | IncF resistance plasmid-to-chromosome coverage ratio | Mobile genetic element dosage |
| `housekeeping_rpoB_norm_depth`| Chromosomal reference gene depth | Single-copy genome control |
| `genome_gc_content` | Isolate genome GC% | Biological taxonomic covariate |
| `sequencing_depth_covariate` | Mean sequencing coverage depth | WGS measurement noise covariate |

---

## 3. Project Structure

```
Tryyyy/
├── README.md                              # Scientific & pipeline documentation
├── AMR_Research_Paper_Draft.md            # Publication-ready research paper (IEEE format)
├── requirements.txt                       # Project dependencies
├── main.py                                # End-to-end execution pipeline
├── data/
│   └── synthetic_amr_genomic_data.csv    # Simulated dataset (3,000 isolates)
└── src/
    ├── __init__.py
    ├── simulate_genomic_data.py          # Continuous genomic feature simulator
    ├── train_amr_model.py                # Baseline Random Forest evaluation
    ├── train_smotetomek_pipeline.py      # Upgraded imblearn SMOTE-Tomek Pipeline (StratifiedKFold)
    ├── train_outbreak_shielded.py        # Outbreak-Shielded Cross-Validation (Regions A & B -> Region C)
    └── clinical_card_reporter.py         # Top-5 feature extraction & CARD API clinical report
```

---

## 4. Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Execution
Run the end-to-end simulation, baseline evaluation, and upgraded SMOTE-Tomek cross-validation:
```bash
python main.py
```

Or execute modular components individually:
```bash
# Generate simulated genomic features
python src/simulate_genomic_data.py

# Train baseline Random Forest (held-out test split)
python src/train_amr_model.py

# Run upgraded imbalanced-learn Pipeline with SMOTE-Tomek (5-Fold StratifiedKFold)
python src/train_smotetomek_pipeline.py
```

---

## 5. Experimental Results: Baseline vs. SMOTE-Tomek Pipeline

Across a rigorous 5-Fold `StratifiedKFold` cross-validation on all 3,000 isolates (150 resistant superbugs, 2,850 susceptible isolates) with **zero validation leakage**:

| Metric (Resistant Superbug Class) | Baseline Random Forest | Upgraded SMOTE-Tomek Pipeline | Relative Impact |
| :--- | :---: | :---: | :---: |
| **Mean Superbug Recall (Sensitivity)** | **16.00% ± 2.49%** | **40.00% ± 6.99%** | **+150.0% Relative Boost (2.50x)** |
| **Mean Superbug Precision (PPV)** | 90.48% ± 7.82% | 54.86% ± 7.44% | Controlled Trade-off |
| **Mean Superbug F1-Score** | 27.01% ± 3.31% | **46.20% ± 7.10%** | **+71.0% F1 Improvement** |
| **Mean Overall Accuracy** | 95.70% | 95.37% | Maintained (>95%) |
| **Total Superbugs Intercepted (TP)** | 24 / 150 (16.0%) | **60 / 150 (40.0%)** | **+36 Additional Superbugs Intercepted** |
| **Total Missed Superbugs (FN)** | 126 / 150 (84.0%) | **90 / 150 (60.0%)** | **28.6% Reduction in False Negatives** |


### Critical Takeaway: The Recall Deficit
- **Overall Accuracy:** `96.33%`
- **Susceptible Recall:** `99.82%`
- **Resistant Superbug Recall:** **`30.00%` (21 missed superbugs / False Negatives)**
- **Clinical Mitigation:** Adjusting the decision threshold from `0.50` down to `0.10` recovers recall to **`86.67%`**, reducing missed superbugs from 21 down to 4.
