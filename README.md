# Data Quality Monitoring and Anomaly Detection System

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Test Suite](https://img.shields.io/badge/pytest-91%20passed-success.svg)]()
[![License](https://img.shields.io/badge/license-Academic%20Open%20Source-green.svg)]()
[![Target Degree](https://img.shields.io/badge/degree-B.Tech%20AI%20%26%20Data%20Science-orange.svg)]()
[![Privacy](https://img.shields.io/badge/privacy-100%25%20In--Memory%20Local-blueviolet.svg)]()

A modular, privacy-preserving, mathematically grounded software suite for tabular data profiling, declarative schema validation, distribution drift monitoring, and unsupervised multivariate anomaly detection.

Designed and developed as a capstone / mini-project for **B.Tech Artificial Intelligence & Data Science (Year 3)**.

---

## 1. Project Overview & Problem Statement

In production Machine Learning (MLOps) and data engineering pipelines, critical failures rarely manifest as explicit runtime exceptions. Instead, pipelines frequently suffer from **"Silent Data Failures"**—scenarios where ETL jobs finish with an exit code of `0` and tables appear fully populated, but the underlying data exhibits subtle corruptions:
- **Syntactic Degradation:** Unannounced null values, duplicate transaction IDs, or boundary limit violations (e.g., negative ages).
- **Distributional Covariate Shift ($P(X_{cur}) \neq P(X_{ref})$):** Production feature distributions drift away from training baselines due to seasonal patterns, demographic shifts, or sensor recalibration, silently degrading machine learning model accuracy without violating data types.
- **Multivariate Outliers:** Observations whose individual features appear normal in univariate isolation, but whose joint multivariate combination represents an impossible or high-risk operational anomaly.

This project implements an end-to-end, local, in-memory diagnostic system that unifies rule-based validation, statistical hypothesis testing, and tree-based machine learning to audit tabular data, score quality, and compile multi-format reports.

---

## 2. Key Features & Capabilities

- **Defensive Ingestion:** Ingests CSV and modern Excel (`.xlsx`) files strictly from memory buffers (`io.BytesIO`) without temporary disk writes. Features interactive Excel worksheet discovery and selection. Legacy `.xls` binary files are rejected.
- **Declarative Schema Validation (Pandera):** Enforces configurable type, boundary, and category rules defined in `config/validation_rules.json`, capturing row-level failure provenance.
- **Transparent Data Quality Score (DQS):** Computes an explainable composite score ($0-100$) dynamically normalized across completeness ($40\%$), uniqueness ($25\%$), and validity ($35\%$).
- **Statistical Drift Monitoring (SciPy / Evidently):** Evaluates continuous variables via the non-parametric **Two-Sample Kolmogorov-Smirnov test** and **Wasserstein-1 distance**; evaluates categorical variables via **Pearson's Chi-Square test** and **Jensen-Shannon divergence**.
- **Multivariate Anomaly Detection (Isolation Forest):** Identifies rare observations via recursive partitioning tree depth $s(x, n) \in [0, 1]$ with robust median imputation and Interquartile Range (IQR) feature deviation explainability.
- **Interactive Streamlit Web Dashboard:** 5 tabbed views with responsive Plotly gauge charts, polar radar diagrams, dual-distribution overlays, and 2D feature projection scatter plots.
- **Enterprise Multi-Format Exports:** Compiles in-memory, styled Excel workbooks (`openpyxl` with Navy theme, freeze panes, and auto-filters), publication-grade academic PDFs (`ReportLab Platypus` with two-pass `NumberedCanvas`), and CSV logs.
- **Reproducible Demo Benchmarks:** Generates deterministic synthetic baseline and corrupted datasets for demonstration and testing.
- **Privacy & Security:** 100% local, offline processing. Zero external network telemetry or third-party cloud analytics persistence.

---

## 3. Technology Stack & Library Responsibilities

| Technology / Library | Version Pinned | Core Responsibility in System |
| :--- | :--- | :--- |
| **Python** | `3.11+` / `3.14` | Core execution runtime. |
| **Pandas** | `==2.3.3` | In-memory DataFrame operations and defensive data manipulation (`copy(deep=True)`). |
| **NumPy** | `==2.5.4` | High-performance numerical computations, array operations, and eCDF derivations. |
| **Pandera** | `==0.35.0` | Vectorized schema definition, declarative business rule enforcement, and failure logs. |
| **Scikit-learn** | `==1.9.1` | Unsupervised `IsolationForest` estimator and `SimpleImputer` for median imputation. |
| **SciPy** | `==1.18.1` | Non-parametric statistical tests (`ks_2samp`, `chisquare`, `wasserstein_distance`). |
| **Evidently** | `==0.7.23` | High-level data drift monitoring and distribution comparison presets. |
| **Streamlit** | `==1.66.1` | Interactive, responsive local web application framework and reactive state management. |
| **Plotly** | `==5.24.1` | Dynamic visualizations: DQS gauge, radar charts, distribution overlays, and anomaly scatter plots. |
| **openpyxl** | `==3.1.5` | Generation and formatting of styled multi-tab Excel workbooks (`.xlsx`). |
| **ReportLab** | `==5.1.0` | In-memory PDF document generation via the Platypus flowable framework. |
| **Pytest** | `==9.1.1` | Automated test suite execution, fixture management, and regression testing. |

---

## 4. System Requirements & Prerequisites

- **Operating System:** Windows 10/11, macOS (12+), or Linux (Ubuntu 20.04+).
- **Python Version:** Python 3.11, 3.12, or 3.14 (tested and verified on Python 3.14.7).
- **RAM:** Minimum 4 GB (8 GB recommended for datasets $> 100,000$ rows).
- **Disk Space:** $\approx 350$ MB for virtual environment and dependencies.

---

## 5. Installation & Setup Instructions

### Step 1: Clone or Navigate to Project
```powershell
cd C:\Users\Admin\.gemini\antigravity\scratch\data-quality-monitor
```

### Step 2: Create a Dedicated Virtual Environment
```powershell
# Create a local virtual environment named .venv
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on Linux / macOS (bash/zsh)
# source .venv/bin/activate
```

### Step 3: Install Pinned Dependencies
```powershell
pip install -r requirements.txt
```

---

## 6. Execution & Verification

### Running the Automated Test Suite (Pytest)
Execute all 91 automated tests across ingestion, profiling, validation, drift, anomaly detection, report generation, and Streamlit integration:

```powershell
# Run the complete test suite
.\.venv\Scripts\pytest.exe -v
```

Expected output:
```text
============================= 91 passed in ~16s =============================
```

### Launching the Streamlit Web Application
To run the interactive web dashboard:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
After execution, open your web browser at:
`http://localhost:8501`

---

## 7. User Guide & Usage Instructions

### 7.1 Using the Synthetic Demo Benchmark
1. In the sidebar, select **"🚀 Load Synthetic Demo Benchmark"**.
2. The system automatically loads pre-configured, deterministic benchmark fixtures (`baseline.csv` and `current.csv`).
3. Explore the 5 tabs to inspect how the system isolates 171 nulls, 18 duplicates, 19 rule violations, distribution shifts on age and transactions, and 8 planted multivariate anomalies.

### 7.2 Uploading Custom CSV or Excel Files
1. In the sidebar, select **"📂 Upload Custom Datasets"**.
2. **Target Dataset:** Upload your production `.csv` or `.xlsx` file. If uploading a multi-sheet Excel file, a dropdown automatically reveals all detected worksheets—select the desired sheet.
3. **Reference Dataset (Optional):** Upload an earlier clean reference dataset or training baseline. Required for distribution and schema drift testing.
4. **Adjust Parameters (Expander):** Modify DQS dimensional weights, drift significance threshold ($\alpha = 0.05$), or Isolation Forest contamination rate ($5\%$).

### 7.3 Downloading Multi-Format Audit Reports
Navigate to **Tab 5: "📑 Reports & Export"**:
- **Academic PDF Audit Report:** Click **"Download PDF Audit Report"** for a formal multi-page document featuring scorecard tables, drift logs, anomaly rankings, and methodology notes.
- **Excel Audit Workbook (.xlsx):** Click **"Download Excel Workbook (.xlsx)"** for a 6-tab spreadsheet with styled navy headers, freeze panes, and column auto-filters.
- **CSV Data Logs:** Download raw CSVs for validation issues, drift test metrics, and flagged anomalies.

---

## 8. Data Quality Score (DQS) Formulation

The composite Data Quality Score ($\text{DQS} \in [0, 100]$) is computed as a weighted linear combination of three fundamental dimensions:

$$\text{DQS} = \frac{\sum_{i \in \mathcal{A}} w_i S_i}{\sum_{i \in \mathcal{A}} w_i}$$

Where $\mathcal{A} \subseteq \{\text{comp}, \text{uniq}, \text{val}\}$ denotes the set of applicable dimensions:
- **Completeness ($S_{\text{comp}}$, default weight $w_1 = 40.0$):** Measures the ratio of non-null, non-empty, and finite cells:
  $$S_{\text{comp}} = (1 - \rho_{\text{miss}}) \times 100, \quad \rho_{\text{miss}} = \frac{\sum \mathbb{I}_{\text{null}}}{N \times M}$$
- **Uniqueness ($S_{\text{uniq}}$, default weight $w_2 = 25.0$):** Measures the proportion of non-duplicate row tuples:
  $$S_{\text{uniq}} = \frac{U}{N} \times 100$$
- **Validity ($S_{\text{val}}$, default weight $w_3 = 35.0$):** Evaluates compliance against configured Pandera boundary and categorical rules:
  $$S_{\text{val}} = \max\left(0, \left(1 - \frac{V_{\text{fail}}}{N_{\text{eval}}}\right) \times 100\right)$$

> **Academic Disclaimer:** DQS is an engineering heuristic indicator defined specifically for this project; it is not an official ISO or IEEE standard. If validation rules are not configured, the validity weight is dynamically omitted and normalized across remaining dimensions rather than artificially penalizing the score.

---

## 9. Known Limitations & Future Roadmap

### Current Limitations:
1. **Large-N $p$-Value Collapse:** With very large sample sizes ($N > 100,000$), standard Kolmogorov-Smirnov $p$-values approach zero even for trivial distribution shifts. (Mitigated by reporting Wasserstein-1 distance).
2. **Tabular Scope:** Designed specifically for 2D tabular formats (CSV, XLSX); unstructured text, audio, and image arrays are not natively parsed.
3. **Median Imputation in Heavy Missingness:** Datasets with $>35\%$ missingness can experience correlation distortion when imputing medians prior to Isolation Forest modeling.

### Future Roadmap:
- **Streaming Ingestion:** Extending batch DataFrame processing to continuous sliding windows on Apache Kafka / Flink event streams.
- **Automated Rule Synthesis:** Integrating historical rule generation (similar to Microsoft Auto-Validate by-History) to infer Pandera contracts automatically.
- **Active Learning Remediation:** Allowing data steward feedback in Streamlit to dynamically fine-tune the Isolation Forest decision boundary.

---

## 10. Project Directory Map

```
data-quality-monitor/
├── app.py                      # Main Streamlit web application entrypoint
├── requirements.txt            # Pinned package versions
├── README.md                   # Comprehensive project documentation
├── PROJECT_SPEC.md             # Detailed engineering specifications & mathematical models
├── DEVELOPMENT_PLAN.md         # Phased implementation plan & verification criteria
├── .gitignore                  # Git ignore rules for virtual environments & caches
│
├── config/
│   └── validation_rules.json   # Declarative schema rules, DQS weights & thresholds
│
├── data/
│   └── demo/                   # Deterministic synthetic benchmark fixtures & docs
│
├── src/
│   ├── data_loader.py          # In-memory CSV/XLSX ingestion and worksheet parser
│   ├── synthetic_generator.py  # Deterministic benchmark data generator
│   ├── profiling.py            # Syntactic profiler (counts, nulls, duplicates)
│   ├── validation.py           # Declarative Pandera schema validation engine
│   ├── quality_score.py        # Composite DQS calculator with weight normalization
│   ├── drift_detection.py      # Statistical drift engine (KS-test, Chi-square, EMD)
│   ├── anomaly_detection.py    # Isolation Forest engine with IQR explainability
│   ├── report_generator.py     # Multi-format export engine (XLSX, PDF, CSV)
│   └── ui_charts.py            # Plotly interactive visualization components
│
├── tests/
│   ├── test_data_loader.py     # Ingestion & parser test suite
│   ├── test_profiling_validation_scoring.py # Profiling, validation & DQS tests
│   ├── test_drift_detection.py # Statistical drift test suite
│   ├── test_anomaly_detection.py # Isolation Forest test suite
│   ├── test_report_generator.py# Excel, PDF & CSV export test suite
│   └── test_app_integration.py # Streamlit pipeline and Plotly integration tests
│
├── reports/                    # Pre-compiled sample export artifacts (PDF, XLSX, CSV)
└── docs/
    ├── RESEARCH_REPORT.md      # IEEE/ACM format academic research paper draft
    ├── VIVA_VOCE_PREP.md       # Comprehensive viva defense guide (25 questions & answers)
    └── project_report_draft.md # Academic degree capstone project report draft
```

---

*Author: B.Tech Artificial Intelligence & Data Science Undergraduate Candidate*  
*Academic Year: 2026*
