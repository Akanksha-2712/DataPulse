"""Deterministic Synthetic Data Generator for Benchmarking & Testing.

Generates realistic, reproducible tabular customer transaction datasets:
1. baseline.csv: A clean, valid reference dataset representing historical normal conditions.
2. current.csv: A second production-batch dataset with controlled, documented data quality
   defects, distribution drift (covariate shift), and multivariate anomalies.

All generation is fully deterministic through explicit NumPy random seeds.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


CATEGORIES = ["Retail", "Travel", "Groceries", "Electronics", "Entertainment", "Utilities"]
ACCOUNT_TYPES = ["Standard", "Premium", "Business"]


def generate_baseline_dataset(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate a clean, high-quality reference/baseline dataset.

    Statistical Profiles (Clean Reference):
    - age: Normal(mean=38, std=9.5) clamped to [18, 75]
    - annual_income: LogNormal(mean=11.0, sigma=0.45) ~ $45,000 - $120,000
    - credit_score: Normal(mean=685, std=55) clamped to [350, 850]
    - transaction_amount: Exponential(scale=110) + $15 (mean ~ $125)
    - debt_to_income: Beta(2.2, 6.0) * 0.70 (range 0.05 - 0.45)
    - category: Discrete choice with stationary market proportions
    - account_type: Discrete choice (60% Standard, 30% Premium, 10% Business)

    Args:
        n_rows: Number of customer transaction rows to synthesize.
        seed: Fixed random seed for complete reproducibility.

    Returns:
        pd.DataFrame: Clean reference dataset.
    """
    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST-{10000 + i:05d}" for i in range(1, n_rows + 1)]

    # Continuous features
    raw_age = rng.normal(loc=38.0, scale=9.5, size=n_rows)
    age = np.clip(np.round(raw_age), 18, 75).astype(int)

    annual_income = np.round(rng.lognormal(mean=11.0, sigma=0.45, size=n_rows), 2)

    raw_credit = rng.normal(loc=685.0, scale=55.0, size=n_rows)
    credit_score = np.clip(np.round(raw_credit), 350, 850).astype(int)

    transaction_amount = np.round(rng.exponential(scale=110.0, size=n_rows) + 15.0, 2)

    debt_to_income = np.round(rng.beta(a=2.2, b=6.0, size=n_rows) * 0.70, 3)

    # Categorical features
    cat_probs = [0.30, 0.15, 0.25, 0.10, 0.10, 0.10]
    category = rng.choice(CATEGORIES, size=n_rows, p=cat_probs)

    acct_probs = [0.60, 0.30, 0.10]
    account_type = rng.choice(ACCOUNT_TYPES, size=n_rows, p=acct_probs)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "transaction_amount": transaction_amount,
        "debt_to_income": debt_to_income,
        "category": category,
        "account_type": account_type,
    })

    return df


def generate_current_dataset(
    n_rows: int = 1200,
    seed: int = 101,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate a drifted and flawed current dataset with documented injected defects.

    Injected Issues:
    1. Missing Values (Completeness):
       - annual_income: 60 missing cells (5.0%)
       - debt_to_income: 48 missing cells (4.0%)
    2. Duplicate Rows (Uniqueness):
       - 18 duplicate rows injected
    3. Rule & Constraint Violations (Validity):
       - age: 6 invalid rows (<18 or >120)
       - credit_score: 6 invalid rows (<300 or >850)
       - annual_income: 5 negative values
       - category: 4 invalid categories ('Unknown_Crypto', 'Illegal_Gambling')
    4. Distributional Drift / Covariate Shift:
       - transaction_amount: Mean shifted from ~$125 to ~$310 (inflation/surge)
       - age: Demographic shift (mean shifted from ~38 to ~49)
       - category: Shift towards 'Travel' (surged from 15% to 35%)
    5. Multivariate ML Outliers / Suspicious Records:
       - 8 specific multivariate anomaly rows violating joint distributions

    Args:
        n_rows: Number of customer transaction rows to synthesize.
        seed: Fixed random seed for complete reproducibility.

    Returns:
        tuple: (pd.DataFrame of current data, dictionary of injected defect metadata)
    """
    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST-{20000 + i:05d}" for i in range(1, n_rows + 1)]

    # 1. Base generation with Covariate Drift
    # Age shifted older: Mean ~49
    raw_age = rng.normal(loc=49.0, scale=11.0, size=n_rows)
    age = np.clip(np.round(raw_age), 18, 80).astype(int)

    annual_income = np.round(rng.lognormal(mean=11.1, sigma=0.50, size=n_rows), 2)

    raw_credit = rng.normal(loc=675.0, scale=60.0, size=n_rows)
    credit_score = np.clip(np.round(raw_credit), 320, 850).astype(int)

    # Transaction amount shifted substantially higher (Covariate Shift: inflation / surge)
    transaction_amount = np.round(rng.exponential(scale=290.0, size=n_rows) + 25.0, 2)

    debt_to_income = np.round(rng.beta(a=2.5, b=5.5, size=n_rows) * 0.75, 3)

    # Categorical shift: Travel surges to 35%, Groceries drops to 12%
    drifted_cat_probs = [0.20, 0.35, 0.12, 0.15, 0.08, 0.10]
    category = rng.choice(CATEGORIES, size=n_rows, p=drifted_cat_probs)

    acct_probs = [0.55, 0.35, 0.10]
    account_type = rng.choice(ACCOUNT_TYPES, size=n_rows, p=acct_probs)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "transaction_amount": transaction_amount,
        "debt_to_income": debt_to_income,
        "category": category,
        "account_type": account_type,
    })

    injected_issues: Dict[str, Any] = {
        "missing_values": {},
        "duplicates_injected": 0,
        "validation_violations": {},
        "distribution_drift": {},
        "ml_anomalies": [],
    }

    # 2. Define reserved violation indices to ensure zero overlap between defect types
    age_violation_indices = [12, 45, 88, 120, 250, 410]
    credit_violation_indices = [22, 67, 134, 189, 305, 512]
    income_violation_indices = [31, 95, 214, 380, 602]
    cat_violation_indices = [73, 156, 421, 789]
    duplicate_indices = set(range(50, 70))

    reserved_for_other_defects = (
        duplicate_indices
        | set(income_violation_indices)
        | set(age_violation_indices)
        | set(credit_violation_indices)
        | set(cat_violation_indices)
    )

    valid_null_candidates = np.array([i for i in range(n_rows) if i not in reserved_for_other_defects])

    # Inject Missing Values (Completeness Defect)
    income_null_indices = rng.choice(valid_null_candidates, size=60, replace=False)
    df.loc[income_null_indices, "annual_income"] = np.nan
    injected_issues["missing_values"]["annual_income"] = 60

    dti_null_indices = rng.choice(valid_null_candidates, size=48, replace=False)
    df.loc[dti_null_indices, "debt_to_income"] = np.nan
    injected_issues["missing_values"]["debt_to_income"] = 48

    # 3. Inject Validation Violations (Validity Defect)
    # Age violations: negative and extreme values
    invalid_ages = [-5, 0, 145, 160, -12, 190]
    for idx, val in zip(age_violation_indices, invalid_ages):
        df.loc[idx, "age"] = val
    injected_issues["validation_violations"]["age"] = len(age_violation_indices)

    # Credit score violations
    invalid_credits = [150, 210, 920, 990, 1050, 240]
    for idx, val in zip(credit_violation_indices, invalid_credits):
        df.loc[idx, "credit_score"] = val
    injected_issues["validation_violations"]["credit_score"] = len(credit_violation_indices)

    # Negative income violations
    invalid_incomes = [-15000.0, -8200.0, -45000.0, -2100.0, -99000.0]
    for idx, val in zip(income_violation_indices, invalid_incomes):
        df.loc[idx, "annual_income"] = val
    injected_issues["validation_violations"]["annual_income_negative"] = len(income_violation_indices)

    # Invalid categorical values
    invalid_cats = ["Unknown_Crypto", "Illegal_Gambling", "Foreign_Bribe", "Unregistered_Asset"]
    for idx, val in zip(cat_violation_indices, invalid_cats):
        df.loc[idx, "category"] = val
    injected_issues["validation_violations"]["category_invalid"] = len(cat_violation_indices)

    # 4. Inject Duplicate Rows (Uniqueness Defect)
    duplicate_source_rows = df.iloc[50:68].copy()
    df = pd.concat([df, duplicate_source_rows], ignore_index=True)
    injected_issues["duplicates_injected"] = len(duplicate_source_rows)

    # 5. Inject Multivariate ML Anomalies (Suspicious Outliers)
    # These satisfy individual univariate boundaries (mostly) but violate multivariate joint logic
    multivariate_anomalies = [
        {
            "customer_id": "ANOM-00001",
            "age": 19,
            "annual_income": 780000.0,
            "credit_score": 420,
            "transaction_amount": 48500.0,
            "debt_to_income": 0.02,
            "category": "Electronics",
            "account_type": "Standard",
            "reason": "Teenager with extreme executive income, terrible credit, massive transaction on basic account.",
        },
        {
            "customer_id": "ANOM-00002",
            "age": 73,
            "annual_income": 14000.0,
            "credit_score": 790,
            "transaction_amount": 62000.0,
            "debt_to_income": 0.98,
            "category": "Travel",
            "account_type": "Standard",
            "reason": "Low-income retiree executing transaction 4x their annual income with near-1.0 DTI.",
        },
        {
            "customer_id": "ANOM-00003",
            "age": 22,
            "annual_income": 28000.0,
            "credit_score": 410,
            "transaction_amount": 39000.0,
            "debt_to_income": 0.94,
            "category": "Retail",
            "account_type": "Business",
            "reason": "Young business account with subprime credit and single transaction exceeding annual wage.",
        },
        {
            "customer_id": "ANOM-00004",
            "age": 68,
            "annual_income": 890000.0,
            "credit_score": 380,
            "transaction_amount": 75000.0,
            "debt_to_income": 0.01,
            "category": "Travel",
            "account_type": "Business",
            "reason": "Top 1% earner with severe default-level credit score and massive single charge.",
        },
        {
            "customer_id": "ANOM-00005",
            "age": 35,
            "annual_income": 32000.0,
            "credit_score": 720,
            "transaction_amount": 85000.0,
            "debt_to_income": 0.99,
            "category": "Electronics",
            "account_type": "Standard",
            "reason": "Average income earner with critical debt burden and enormous retail transaction.",
        },
        {
            "customer_id": "ANOM-00006",
            "age": 20,
            "annual_income": 18000.0,
            "credit_score": 840,
            "transaction_amount": 42000.0,
            "debt_to_income": 0.91,
            "category": "Travel",
            "account_type": "Premium",
            "reason": "Near-perfect 840 credit score at age 20 with poverty-level income and luxury transactions.",
        },
        {
            "customer_id": "ANOM-00007",
            "age": 55,
            "annual_income": 950000.0,
            "credit_score": 850,
            "transaction_amount": 120000.0,
            "debt_to_income": 0.01,
            "category": "Utilities",
            "account_type": "Business",
            "reason": "Six-figure utility bill payment even for a business account.",
        },
        {
            "customer_id": "ANOM-00008",
            "age": 42,
            "annual_income": 22000.0,
            "credit_score": 460,
            "transaction_amount": 56000.0,
            "debt_to_income": 0.88,
            "category": "Entertainment",
            "account_type": "Standard",
            "reason": "Low-wage earner spending $56k on entertainment.",
        },
    ]

    anomaly_df = pd.DataFrame(multivariate_anomalies)
    # Drop the explanation column before merging into dataset
    clean_anomaly_df = anomaly_df.drop(columns=["reason"])
    df = pd.concat([df, clean_anomaly_df], ignore_index=True)

    injected_issues["ml_anomalies"] = multivariate_anomalies

    # Document documented distribution changes
    injected_issues["distribution_drift"] = {
        "transaction_amount": "Exponential scale changed 110 -> 290, loc 15 -> 25",
        "age": "Normal loc shifted 38 -> 49, scale 9.5 -> 11.0",
        "category": "Travel proportion surged from 15% to 35%",
    }

    return df, injected_issues


def create_demo_fixtures(
    output_dir: Path | str = "data/demo",
    seed_baseline: int = 42,
    seed_current: int = 101,
) -> Dict[str, str]:
    """Generate and write the standard demo CSV and XLSX benchmark files to disk.

    Creates:
    - baseline.csv: Clean reference dataset.
    - current.csv: Corrupted and drifted production dataset.
    - multi_sheet_demo.xlsx: Multi-sheet workbook with Baseline and Current sheets.
    - DEMO_DATA_DOCUMENTATION.md: Detailed markdown log of injected defects.

    Args:
        output_dir: Target directory path.
        seed_baseline: Fixed seed for baseline data.
        seed_current: Fixed seed for current data.

    Returns:
        dict: File paths of generated fixtures.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    baseline_df = generate_baseline_dataset(n_rows=1200, seed=seed_baseline)
    current_df, issues = generate_current_dataset(n_rows=1200, seed=seed_current)

    baseline_path = out_path / "baseline.csv"
    current_path = out_path / "current.csv"
    excel_path = out_path / "multi_sheet_demo.xlsx"
    doc_path = out_path / "DEMO_DATA_DOCUMENTATION.md"

    # Save CSVs
    baseline_df.to_csv(baseline_path, index=False)
    current_df.to_csv(current_path, index=False)

    # Save multi-sheet Excel
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        baseline_df.to_excel(writer, sheet_name="Baseline", index=False)
        current_df.to_excel(writer, sheet_name="Current", index=False)

    # Save Documentation of Injected Issues
    doc_content = f"""# Demo Benchmark Datasets: Documentation of Injected Defects

**Generated Files:**
- `baseline.csv` ({len(baseline_df)} rows, 8 columns) - Clean reference dataset.
- `current.csv` ({len(current_df)} rows, 8 columns) - Flawed and drifted production batch.
- `multi_sheet_demo.xlsx` - Multi-sheet workbook containing both 'Baseline' and 'Current'.

---

## 1. Deliberately Injected Defects in `current.csv`

### A. Missing Values (Completeness Defect)
- `annual_income`: {issues['missing_values'].get('annual_income', 0)} missing values (`np.nan`, 5.0%)
- `debt_to_income`: {issues['missing_values'].get('debt_to_income', 0)} missing values (`np.nan`, 4.0%)

### B. Duplicate Records (Uniqueness Defect)
- Injected: {issues.get('duplicates_injected', 0)} exact duplicate rows (rows copied from indices 50..67).

### C. Validation Violations (Domain Validity Defect)
- `age`: 6 rows with invalid values (`[-5, 0, 145, 160, -12, 190]`).
- `credit_score`: 6 rows with invalid values (`[150, 210, 920, 990, 1050, 240]`).
- `annual_income`: 5 rows with negative income (`[-15000.0, -8200.0, -45000.0, -2100.0, -99000.0]`).
- `category`: 4 rows with invalid categorical values (`'Unknown_Crypto'`, `'Illegal_Gambling'`, `'Foreign_Bribe'`, `'Unregistered_Asset'`).

### D. Distributional Covariate Shift (Statistical Drift)
- `transaction_amount`: Mean shifted from ~$125 in Baseline to ~$310 in Current (simulating inflation / consumer spending surge).
- `age`: Mean shifted from ~38 years to ~49 years (demographic shift).
- `category`: 'Travel' proportion surged from 15% in Baseline to 35% in Current.

### E. Multivariate ML Anomalies (Suspicious Records)
{len(issues['ml_anomalies'])} multivariate outlier records injected (IDs `ANOM-00001` to `ANOM-00008`):
"""
    for anom in issues["ml_anomalies"]:
        doc_content += f"- **{anom['customer_id']}**: {anom['reason']}\n"

    doc_path.write_text(doc_content, encoding="utf-8")

    return {
        "baseline_csv": str(baseline_path),
        "current_csv": str(current_path),
        "multi_sheet_excel": str(excel_path),
        "documentation": str(doc_path),
    }
