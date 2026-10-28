# ⚡ DataPulse — Keep Your Data Clean. Reliable. Ready.

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Test Suite](https://img.shields.io/badge/pytest-96%20passed-success.svg)]()
[![Framework](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)
[![Privacy](https://img.shields.io/badge/privacy-100%25%20In--Memory%20Local-blueviolet.svg)]()

**DataPulse** is an intelligent tabular data quality monitoring and anomaly detection platform that helps you understand, validate, and improve your data. 

Upload your dataset to detect missing values, duplicates, data inconsistencies, statistical drift, and unusual multivariate patterns—all through a clean, interactive dashboard.

> ***Your data. Our insights. Better decisions.***

---

## 🚀 Key Features

* **Memory-Safe Ingestion:** Reads CSV and modern Excel (`.xlsx`) files directly into memory buffers (`io.BytesIO`) without temporary disk writes. Features automatic worksheet discovery for multi-tab workbooks.
* **Syntactic Data Profiling:** Computes missing cell percentages, duplicate row counts, empty strings, and memory footprints.
* **Declarative Schema Validation (Pandera):** Enforces strict boundary, range, and categorical rules defined in `config/validation_rules.json`, capturing exact row references and severity levels (Critical, Error, Warning).
* **Transparent Data Quality Score (DQS):** Computes a composite health indicator ($0-100$) dynamically weighted across completeness ($40\%$), uniqueness ($25\%$), and validity ($35\%$).
* **Statistical Covariate Shift Monitoring:** Measures population drift against a reference baseline using the **Two-Sample Kolmogorov-Smirnov test** (continuous features) and **Pearson's Chi-Square test** (categorical features).
* **Unsupervised Multivariate Anomaly Detection:** Employs **Scikit-learn Isolation Forest** to isolate multi-feature outliers near partition roots with continuous suspicion scores ($0-1$) and Interquartile Range (IQR) feature deviation explainability.
* **1-Click Multi-Format Reporting:** Generates publication-ready academic PDFs (`ReportLab Platypus` with dynamic "Page X of Y" page numbering), 6-tab styled Excel workbooks (`openpyxl` with navy headers, freeze panes, and auto-filters), and raw CSV logs.
* **Privacy & Security:** 100% in-memory processing. Zero disk persistence or third-party cloud telemetry.

---

## 📁 Repository Structure

```text
DataPulse/
├── app.py                      # Main Streamlit Web Application
├── requirements.txt            # Pinned dependencies (Python 3.10+)
├── Dockerfile                  # Container deployment specification
├── README.md                   # Project documentation
│
├── .streamlit/
│   └── config.toml             # Streamlit cloud theme & server settings
│
├── config/
│   └── validation_rules.json   # Pandera declarative schema validation rules
│
├── data/demo/                  # Offline Benchmark Datasets
│   ├── current.csv             # 1,226 target records with controlled defects
│   ├── baseline.csv            # 1,200 clean baseline records
│   └── multi_sheet_demo.xlsx   # Multi-tab workbook ('Baseline' & 'Current' sheets)
│
├── src/                        # Core Python Backend Modules
│   ├── __init__.py
│   ├── anomaly_detection.py    # Isolation Forest multivariate outlier detection
│   ├── data_loader.py          # Memory-safe CSV/XLSX loader & sheet picker
│   ├── drift_detection.py      # Kolmogorov-Smirnov & Chi-Square drift engine
│   ├── profiling.py            # Syntactic data profiling (nulls, duplicates)
│   ├── quality_score.py        # Composite Data Quality Score (DQS) algorithm
│   ├── report_generator.py     # openpyxl Excel & ReportLab PDF generators
│   ├── synthetic_generator.py  # Deterministic benchmark data generator
│   ├── ui_charts.py            # Plotly interactive charts & scatter projections
│   └── validation.py           # Pandera schema contract enforcement
│
└── tests/                      # Automated Test Suite (96 Tests)
    ├── test_anomaly_detection.py
    ├── test_app_integration.py
    ├── test_data_loader.py
    ├── test_drift_detection.py
    ├── test_profiling_validation_scoring.py
    ├── test_report_generator.py
    └── test_ui_workflow.py
```

---

## 🛠️ Quickstart & Local Setup

### 1. Clone the Repository
```powershell
git clone https://github.com/Akanksha-2712/DataPulse.git
cd DataPulse
```

### 2. Create and Activate Virtual Environment
```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Launch the Web Application
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Running Automated Tests

Run the complete test suite across all 7 test modules:

```powershell
pytest -q
```
*Expected: 96 passed.*

---

## 🌐 Deploy to Streamlit Community Cloud

1. Push this repository to your GitHub account.
2. Sign in to **[share.streamlit.io](https://share.streamlit.io)** with GitHub.
3. Click **"New app"**, select your repository `Akanksha-2712/DataPulse`, set branch to `main`, and main file to `app.py`.
4. Click **"Deploy!"**.
