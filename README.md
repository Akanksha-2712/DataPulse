# Data Quality Monitoring and Anomaly Detection System

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-Stage%201%20Scaffolding%20Complete-success.svg)]()
[![Target Degree](https://img.shields.io/badge/degree-B.Tech%20AI%20%26%20Data%20Science-orange.svg)]()

A modular, offline-first, mathematically grounded system for tabular data quality assessment, schema validation, distributional drift monitoring, and unsupervised multivariate anomaly detection.

Developed as a mini-project / capstone prototype for **B.Tech Artificial Intelligence & Data Science (Year 3)**.

---

## 1. Project Overview & Core Capabilities

Data pipelines and machine learning models often degrade silently without throwing explicit software exceptions. This system addresses silent data degradation across three distinct analytical dimensions:

1. **Syntactic & Schema Integrity:** Missing value profiling, duplicate row identification, data type verification, and declarative constraint validation using **Pandera**.
2. **Distributional Covariate Shift (Drift):** Comparing reference baseline datasets with current production batches using **Evidently** and **SciPy** (Two-sample Kolmogorov-Smirnov test, Wasserstein distance, Pearson's Chi-Square test).
3. **Multivariate Anomaly Isolation:** Unsupervised detection of subtle, unusual observations using **Scikit-learn Isolation Forest** ($s(x, n)$ path length scoring).
4. **Transparent Quality Scoring:** Deterministic **Data Quality Score (DQS)** ranging from $0$ to $100$ with documented weighting and mathematical transparency.
5. **Multi-Format Audit Reporting:** Interactive **Streamlit** dashboard with **Plotly** graphics, plus downloadable audits in **CSV**, styled **openpyxl XLSX**, and formal **ReportLab PDF**.

---

## 2. Directory Structure

```
data-quality-monitor/
├── app.py                      # Main Streamlit web application entrypoint
├── requirements.txt            # Pinned dependencies
├── README.md                   # Project overview and setup instructions
├── PROJECT_SPEC.md             # Detailed engineering specifications & mathematical models
├── DEVELOPMENT_PLAN.md         # Phased implementation plan & viva preparation
├── .gitignore                  # Git ignore rules for virtual environments, temp & reports
│
├── config/                     # Configuration and rule sets
│   └── validation_rules.json   # Declarative schema rules, DQS weights & thresholds
│
├── data/                       # Datasets
│   └── demo/                   # Deterministic synthetic benchmark fixtures
│
├── src/                        # Core application source modules
│   └── __init__.py             # Package marker
│
├── tests/                      # Automated test suite (Pytest)
│   └── __init__.py             # Test package marker
│
├── reports/                    # Generated audit outputs (PDF, XLSX, CSV)
└── docs/                       # Research report drafts & viva defense guides
```

---

## 3. Technology Stack

- **Runtime:** Python
- **Tabular Data Processing:** Pandas & NumPy
- **Schema Validation:** Pandera (vectorized DataFrame schemas)
- **Statistical Drift:** Evidently & SciPy (Two-sample K-S, Chi-square, Wasserstein)
- **Anomaly Detection:** Scikit-learn (Isolation Forest)
- **Dashboard Interface:** Streamlit & Plotly
- **Workbook & Document Generation:** openpyxl (Excel) & ReportLab (PDF)
- **Quality Assurance:** Pytest

---

## 4. Setup & Installation

### Step 1: Create Virtual Environment
```powershell
# From the project root (data-quality-monitor/)
python -m venv .venv

# Activate the virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 3: Verify Installation
```powershell
python -c "import pandas, numpy, pandera, scikit_learn, streamlit, plotly, openpyxl, reportlab, scipy, pytest; print('All core dependencies imported successfully!')"
```

---

## 5. Architectural Principles & Viva Highlights

- **Data Immutability:** Input datasets are treated strictly as read-only buffers. Deep copies prevent side-effects.
- **ML Anomaly $\neq$ Defective Record:** Isolation Forest flags statistically unusual observations in multivariate feature space. These are highlighted as suspicious records for human review, not deleted.
- **Data Drift $\neq$ Pipeline Error:** Distributional shifts reflect real-world environment changes (covariate shift), signaling the need for model recalibration rather than software failure.
- **Zero External Telemetry:** All data processing, statistical tests, and machine learning inferences execute 100% locally on the host machine.
