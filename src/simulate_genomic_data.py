"""
Bacterial AMR Genomic Feature Simulator with Geographic Outbreak Modeling
==========================================================================
Simulates realistic, continuous-valued genomic features from whole-genome
sequencing (WGS) data for bacterial pathogens across distinct mock geographic
regions (Region A, Region B, and Region C).

Bioinformatics & Epidemiological Context:
-----------------------------------------
Clonal expansion causes bacterial strains within the same hospital outbreak
or geographic region to share correlated genetic lineages:
- Region A (1050 isolates): Outbreak dominated by KPC-producing plasmids (bla_KPC, IncF).
- Region B (1050 isolates): Outbreak dominated by NDM metallo-beta-lactamase & AcrAB efflux.
- Region C (900 isolates): Outbreak dominated by OXA-48 carbapenemase combined with severe
  outer-membrane porin permeability defects (OmpK35 / OmpK36 loss).

This geographic structure enables the 'Outbreak-Shielded' cross-validation strategy,
where models trained on Regions A & B are evaluated out-of-distribution on unseen Region C.

Class Distribution:
-------------------
- Exactly 95.0% Susceptible (2,850 isolates)
- Exactly 5.0% Resistant Superbugs (150 isolates)
"""

import numpy as np
import pandas as pd


def simulate_amr_genomic_dataset(
    n_samples: int = 3000,
    resistant_prevalence: float = 0.05,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Simulates continuous bacterial genomic features with geographic regions.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Geographic Partition Allocations
    # Region A: 1050 isolates (~35%)
    # Region B: 1050 isolates (~35%)
    # Region C: 900 isolates  (~30%)
    n_reg_a = 1050
    n_reg_b = 1050
    n_reg_c = 900

    regions = np.array(["Region_A"] * n_reg_a + ["Region_B"] * n_reg_b + ["Region_C"] * n_reg_c)

    # Regional resistant counts strictly calibrated to total 150 (5.0%)
    # Region A: 52 resistant, 998 susceptible
    # Region B: 53 resistant, 997 susceptible
    # Region C: 45 resistant, 855 susceptible (exactly 5.0%)
    res_a, susc_a = 52, 998
    res_b, susc_b = 53, 997
    res_c, susc_c = 45, 855

    # 2. Continuous Genomic Features by Region (incorporating clonal lineage signatures)
    def generate_regional_cohort(n_susc, n_res, region_name):
        n_total = n_susc + n_res
        
        # Base background features for all isolates in the region
        if region_name == "Region_A":
            # KPC-dominant clonal lineage
            bla_kpc = np.concatenate([
                rng.exponential(scale=0.15, size=n_susc),
                rng.gamma(shape=4.2, scale=1.4, size=n_res) + 1.5
            ])
            bla_ndm = np.concatenate([
                rng.exponential(scale=0.08, size=n_susc),
                rng.exponential(scale=0.50, size=n_res)
            ])
            bla_ctx = np.concatenate([
                rng.exponential(scale=0.80, size=n_susc),
                rng.gamma(shape=3.0, scale=1.1, size=n_res) + 0.8
            ])
            bla_oxa = np.concatenate([
                rng.exponential(scale=0.06, size=n_susc),
                rng.exponential(scale=0.40, size=n_res)
            ])
            efflux = np.concatenate([
                rng.normal(loc=1.5, scale=0.4, size=n_susc),
                rng.normal(loc=2.2, scale=0.6, size=n_res)
            ])
            porin_k35 = np.concatenate([
                rng.normal(loc=1.2, scale=0.4, size=n_susc),
                rng.normal(loc=1.8, scale=0.6, size=n_res)
            ])
            porin_k36 = np.concatenate([
                rng.normal(loc=1.3, scale=0.4, size=n_susc),
                rng.normal(loc=1.9, scale=0.6, size=n_res)
            ])
            incf_plasmid = np.concatenate([
                rng.exponential(scale=0.20, size=n_susc),
                rng.gamma(shape=3.5, scale=1.2, size=n_res) + 1.2
            ])
            gc_content = rng.normal(loc=53.1, scale=1.0, size=n_total)

        elif region_name == "Region_B":
            # NDM & AcrAB-TolC efflux-dominant lineage
            bla_kpc = np.concatenate([
                rng.exponential(scale=0.10, size=n_susc),
                rng.exponential(scale=0.60, size=n_res)
            ])
            bla_ndm = np.concatenate([
                rng.exponential(scale=0.07, size=n_susc),
                rng.gamma(shape=4.0, scale=1.3, size=n_res) + 1.2
            ])
            bla_ctx = np.concatenate([
                rng.exponential(scale=0.70, size=n_susc),
                rng.gamma(shape=2.5, scale=1.0, size=n_res) + 0.5
            ])
            bla_oxa = np.concatenate([
                rng.exponential(scale=0.08, size=n_susc),
                rng.exponential(scale=0.50, size=n_res)
            ])
            efflux = np.concatenate([
                rng.normal(loc=1.6, scale=0.4, size=n_susc),
                rng.normal(loc=4.5, scale=1.1, size=n_res)  # Strong efflux
            ])
            porin_k35 = np.concatenate([
                rng.normal(loc=1.4, scale=0.4, size=n_susc),
                rng.normal(loc=2.2, scale=0.7, size=n_res)
            ])
            porin_k36 = np.concatenate([
                rng.normal(loc=1.4, scale=0.4, size=n_susc),
                rng.normal(loc=2.0, scale=0.6, size=n_res)
            ])
            incf_plasmid = np.concatenate([
                rng.exponential(scale=0.25, size=n_susc),
                rng.gamma(shape=2.2, scale=1.0, size=n_res) + 0.6
            ])
            gc_content = rng.normal(loc=53.4, scale=1.1, size=n_total)

        else:  # Region C (Unseen Outbreak: OXA-48 + severe porin loss)
            bla_kpc = np.concatenate([
                rng.exponential(scale=0.12, size=n_susc),
                rng.exponential(scale=0.50, size=n_res)
            ])
            bla_ndm = np.concatenate([
                rng.exponential(scale=0.06, size=n_susc),
                rng.exponential(scale=0.40, size=n_res)
            ])
            bla_ctx = np.concatenate([
                rng.exponential(scale=0.85, size=n_susc),
                rng.gamma(shape=2.8, scale=1.1, size=n_res) + 0.7
            ])
            bla_oxa = np.concatenate([
                rng.exponential(scale=0.09, size=n_susc),
                rng.gamma(shape=3.8, scale=1.2, size=n_res) + 1.1  # High OXA-48
            ])
            efflux = np.concatenate([
                rng.normal(loc=1.5, scale=0.4, size=n_susc),
                rng.normal(loc=2.4, scale=0.7, size=n_res)
            ])
            porin_k35 = np.concatenate([
                rng.normal(loc=1.3, scale=0.4, size=n_susc),
                rng.normal(loc=4.8, scale=1.2, size=n_res)  # Severe porin loss
            ])
            porin_k36 = np.concatenate([
                rng.normal(loc=1.3, scale=0.4, size=n_susc),
                rng.normal(loc=4.6, scale=1.1, size=n_res)  # Severe porin loss
            ])
            incf_plasmid = np.concatenate([
                rng.exponential(scale=0.18, size=n_susc),
                rng.gamma(shape=2.5, scale=1.1, size=n_res) + 0.8
            ])
            gc_content = rng.normal(loc=52.9, scale=1.2, size=n_total)

        # Fluoroquinolone mutations & sequencing covariates
        gyra_vaf = np.concatenate([
            rng.beta(0.2, 5.0, size=n_susc),
            rng.beta(3.5, 1.2, size=n_res)
        ])
        parc_vaf = np.concatenate([
            rng.beta(0.2, 5.0, size=n_susc),
            rng.beta(2.8, 1.5, size=n_res)
        ])
        rpob_depth = rng.normal(loc=1.00, scale=0.08, size=n_total)
        seq_depth = rng.normal(loc=65.0, scale=13.0, size=n_total)
        labels = np.array([0] * n_susc + [1] * n_res)

        return pd.DataFrame({
            "geographic_region": region_name,
            "bla_KPC_copy_number": np.round(np.clip(bla_kpc, 0.0, None), 3),
            "bla_NDM_copy_number": np.round(np.clip(bla_ndm, 0.0, None), 3),
            "bla_CTX_M_copy_number": np.round(np.clip(bla_ctx, 0.0, None), 3),
            "bla_OXA_48_copy_number": np.round(np.clip(bla_oxa, 0.0, None), 3),
            "acrAB_tolC_efflux_score": np.round(np.clip(efflux, 0.2, None), 3),
            "ompK35_porin_loss_index": np.round(np.clip(porin_k35, 0.0, None), 3),
            "ompK36_porin_loss_index": np.round(np.clip(porin_k36, 0.0, None), 3),
            "gyrA_S83L_vaf": np.round(np.clip(gyra_vaf, 0.0, 1.0), 4),
            "parC_S80I_vaf": np.round(np.clip(parc_vaf, 0.0, 1.0), 4),
            "incF_plasmid_copy_ratio": np.round(np.clip(incf_plasmid, 0.0, None), 3),
            "housekeeping_rpoB_norm_depth": np.round(rpob_depth, 3),
            "genome_gc_content": np.round(gc_content, 2),
            "sequencing_depth_covariate": np.round(seq_depth, 1),
            "amr_phenotype": labels
        })

    df_a = generate_regional_cohort(susc_a, res_a, "Region_A")
    df_b = generate_regional_cohort(susc_b, res_b, "Region_B")
    df_c = generate_regional_cohort(susc_c, res_c, "Region_C")

    # Combine cohorts
    df = pd.concat([df_a, df_b, df_c], ignore_index=True)
    sample_ids = [f"ISO_{i+1:05d}" for i in range(len(df))]
    df.insert(0, "isolate_id", sample_ids)

    # Shuffle within regions but keep regional integrity
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating simulated AMR bacterial genomics dataset with geographic regions...")
    df = simulate_amr_genomic_dataset(n_samples=3000, resistant_prevalence=0.05, random_seed=42)
    output_path = "data/synthetic_amr_genomic_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully created and saved to '{output_path}'.")
    print(f"Total isolates: {len(df)}")
    counts = df["amr_phenotype"].value_counts()
    print(f"Overall Susceptible (0): {counts[0]} ({counts[0] / len(df) * 100:.2f}%)")
    print(f"Overall Resistant   (1): {counts[1]} ({counts[1] / len(df) * 100:.2f}%)")
    print("\nRegional Distribution:")
    print(pd.crosstab(df["geographic_region"], df["amr_phenotype"], margins=True))
