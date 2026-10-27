# Project Report: Data Quality Monitoring and Anomaly Detection System

**Degree:** Bachelor of Technology (B.Tech) in Artificial Intelligence and Data Science  
**Document Type:** Capstone Academic Project Report Draft  
**Target Submission:** Department of Artificial Intelligence & Data Science  
**Academic Year:** 2025–2026  

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Aim and Objectives](#4-aim-and-objectives)
5. [Literature Review](#5-literature-review)
6. [Existing System and Limitations](#6-existing-system-and-limitations)
7. [Proposed System](#7-proposed-system)
8. [Requirements and Technology Stack](#8-requirements-and-technology-stack)
9. [System Architecture](#9-system-architecture)
10. [Methodology and Algorithms](#10-methodology-and-algorithms)
11. [Data Quality Score Formula](#11-data-quality-score-formula)
12. [Implementation Details](#12-implementation-details)
13. [Testing and Experimental Setup](#13-testing-and-experimental-setup)
14. [Results and Discussion](#14-results-and-discussion)
15. [Limitations](#15-limitations)
16. [Conclusion and Future Scope](#16-conclusion-and-future-scope)
17. [References](#17-references)

---

## 1. Abstract

Modern machine learning (ML) workflows and data-driven analytical pipelines are exceptionally vulnerable to degraded input data. Unlike traditional software systems where malformed inputs typically trigger runtime crashes or unhandled exceptions, data pipelines frequently experience "silent data degradation." In such scenarios, pipelines complete without raising software errors, yet downstream machine learning models suffer from severe performance deterioration due to subtle syntactic defects, population distribution shifts, or multivariate anomalies.

This project designs and implements an end-to-end, privacy-preserving, and explainable **Data Quality Monitoring and Anomaly Detection System** tailored for tabular datasets (CSV and XLSX formats). The system integrates three complementary layers: (1) deterministic declarative schema validation using Pandera, (2) non-parametric statistical distribution drift monitoring using SciPy (Two-Sample Kolmogorov-Smirnov test and Pearson's Chi-Square test), and (3) unsupervised multivariate anomaly detection using Scikit-learn's Isolation Forest algorithm. 

To provide an executive-level summary without concealing underlying issues, a transparent composite **Data Quality Score (DQS)** is formulated with dynamic weight normalization across completeness ($40\%$), uniqueness ($25\%$), and validity ($35\%$). The entire system operates in-memory with zero cloud persistence, features an interactive Streamlit dashboard with Plotly visual diagnostics, and compiles multi-format audit artifacts (openpyxl styled Excel workbooks and ReportLab Platypus PDFs). The implementation is verified via a 91-test automated Pytest suite. A structured experimental protocol and template are provided to evaluate detection precision, recall, and processing overhead on controlled benchmark datasets.

---

## 2. Introduction

Data is the fundamental fuel of modern artificial intelligence and decision-support systems. In commercial applications—ranging from credit scoring and algorithmic fraud detection to medical diagnostics and predictive maintenance—the accuracy, reliability, and fairness of machine learning models depend directly on the quality of the ingested training and inference data.

Historically, organizations treated data validation as an ad-hoc preprocessing task, typically implemented as sporadic SQL queries or disconnected Python scripts checking for null values. However, as organizations transition from static experimental models to continuous production pipelines (MLOps), manual checks become unsustainable. When input data changes subtly—a phenomenon known as data downtime or data decay—predictive models experience performance degradation without raising any operational alarms.

To maintain robust machine learning pipelines, a modern data quality assurance framework must simultaneously monitor:
1. **Syntactic integrity:** Confirming that the data structure, types, nullness, and uniqueness comply with schema expectations.
2. **Semantic validity:** Ensuring that individual values obey domain-specific boundaries and relational business rules.
3. **Statistical stationarity:** Detecting whether the empirical probability distribution of incoming production data has drifted relative to historical baseline distributions (covariate shift).
4. **Multivariate consistency:** Identifying complex observations that appear benign in univariate isolation but represent severe anomalies when examined across multiple correlated dimensions.

This project addresses these needs by establishing an integrated, local, and explainable software platform designed to give data practitioners immediate visibility into tabular data health.

---

## 3. Problem Statement

Organizations increasingly rely on automated data pipelines to feed production machine learning models and analytical dashboards. However, existing data validation workflows suffer from three major systemic vulnerabilities:

1. **The Silent Failure Dilemma:** Traditional software engineering assertions verify process completion (e.g., successful API response, non-empty database table) rather than data health. When upstream data changes—such as a changed third-party schema, corrupted sensor calibrations, or altered user demographics—the pipeline executes successfully, leading to "garbage in, garbage out" without warning.
2. **The Covariate Shift Blind Spot:** Standard schema validators evaluate data row-by-row against static thresholds. They are blind to population-level statistical distribution shifts (covariate shift) where no single row violates an explicit rule, but the overall probability density function $P(X)$ changes significantly, invalidating downstream ML model assumptions.
3. **The Multivariate Outlier Blind Spot:** Univariate range checks (e.g., verifying that `Age` is between 18 and 100, and `Transaction_Amount` is between \$0 and \$100,000) fail to detect multivariate anomalies where each feature is individually within valid ranges, but their joint occurrence (e.g., an 18-year-old student with a \$90,000 retail transaction and subprime credit) represents a critical operational risk.

Furthermore, many commercial monitoring tools require uploading proprietary enterprise data to third-party cloud SaaS platforms, violating regulatory privacy constraints (e.g., GDPR, HIPAA, financial data compliance). There is an acute academic and operational need for a transparent, 100% in-memory, local system that unifies rule validation, statistical drift testing, and multivariate anomaly detection with verifiable explainability.

---

## 4. Aim and Objectives

### 4.1 Primary Aim
To design, implement, and evaluate a modular, privacy-preserving, and mathematically grounded Data Quality Monitoring and Anomaly Detection System capable of diagnosing tabular data defects, quantifying distributional drift, and isolating multivariate anomalies.

### 4.2 Specific Objectives
1. **Robust In-Memory Ingestion:** Develop defensive parsers for CSV and modern multi-sheet XLSX workbooks that operate strictly in volatile memory, rejecting unsafe legacy formats (`.xls`) and preventing source data corruption.
2. **Declarative Rule Validation:** Implement a schema validation engine using Pandera to enforce declarative boundaries, categorical constraints, and non-null rules, outputting structured violation logs with exact row indices.
3. **Transparent Quality Scoring:** Formulate an explainable Data Quality Score (DQS) with dynamic weight normalization across completeness ($40\%$), uniqueness ($25\%$), and validity ($35\%$).
4. **Distributional Drift Monitoring:** Construct a statistical two-sample testing engine comparing baseline reference datasets with current batches using the Two-Sample Kolmogorov-Smirnov test and Pearson's Chi-Square test, supplemented by the Wasserstein-1 metric.
5. **Unsupervised Anomaly Detection:** Implement Scikit-learn's Isolation Forest with robust median imputation to isolate complex multivariate outliers, generating normalized suspicion scores and Interquartile Range (IQR) feature deviation explanations.
6. **Interactive Visualization & Multi-Format Reporting:** Build a local Streamlit dashboard with Plotly visual diagnostics and in-memory export generators for styled openpyxl Excel workbooks, ReportLab Platypus academic PDFs, and CSV logs.
7. **Automated Verification:** Construct a comprehensive automated Pytest suite verifying edge cases, mathematical boundary conditions, and degraded operating modes with $>80\%$ test coverage.

---

## 5. Literature Review

The development of automated data quality systems for machine learning is informed by several foundational research efforts across academia and industry:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                   FOUNDATIONAL LITERATURE TAXONOMY                     │
  └────────────────────────────────────────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  [ Production ML Systems ]     [ Distributed DQ Engines ]   [ Surveys & Frameworks ]
  - Breck et al. (Google)       - Schelter et al. (Amazon)   - Ehrlinger et al. (2022)
    TensorFlow Data Validation    Deequ Unit Tests for Data    Tool taxonomy & metrics
  - Zhang et al. (Microsoft)                                 - Budach et al. (2024)
    Auto-Validate by-History                                   ML dataset quality survey
```

### 5.1 Data Validation for Machine Learning (Breck et al., Google, 2019)
Breck et al. [1] investigated data validation challenges within production machine learning pipelines at Google, establishing the **TensorFlow Data Validation (TFDV)** framework. The authors identified that ML systems are uniquely prone to data-specific failures that cannot be caught by conventional unit tests. They formalized three core categories of data anomalies:
- **Schema Skew:** Discrepancies between expected and actual data structures (e.g., unexpected missing columns, integer-to-float conversions).
- **Distribution Skew (Drift):** Discrepancies between the distribution of features used during training and those encountered during production serving.
- **Training-Serving Skew:** Systematic differences caused by divergent transformations in production scoring versus batch training pipelines.

Breck et al. argued that production data validation must automatically infer schemas, validate streaming and batch data against stored schema statistics, and measure distribution distance using metrics like the Chebyshev distance and Jensen-Shannon divergence.  
*Distinction in Our Work:* While TFDV is tightly integrated into the TensorFlow ecosystem and optimized for Protobuf / Apache Beam distributed pipelines, our system focuses on general-purpose tabular data (CSV/XLSX) in lightweight Python/Pandas environments, using Pandera for schema enforcement and SciPy for non-parametric statistical hypothesis testing.

### 5.2 Automating Large-Scale Data Quality Verification (Schelter et al., Amazon / VLDB, 2018)
Schelter et al. [2] introduced **Deequ**, an open-source library built atop Apache Spark for unit testing large-scale enterprise datasets at Amazon. The authors highlighted that while software engineers routinely write unit tests for application code, data engineers lacked equivalent mechanisms to write "unit tests for data." Deequ formalizes data verification as a set of declarative constraints (e.g., `hasCompleteness`, `isUnique`, `hasCorrelation`) executed via single-pass distributed aggregations.

Key findings from Schelter et al. include:
- Manual rule authoring is laborious; systems should provide automated constraint suggestion based on historical data profiling.
- Computing exact metrics over massive datasets requires incremental state updates to avoid repeated full-table scans.  
*Distinction in Our Work:* Deequ is implemented in Scala/Spark for distributed cluster environments. Our project adapts the concept of declarative constraint verification to local tabular data using Pandera, specifically integrating unsupervised machine learning (Isolation Forest) which Deequ does not natively provide as an integrated layer.

### 5.3 A Survey of Data Quality Measurement and Monitoring Tools (Ehrlinger et al., Frontiers in Big Data, 2022)
Ehrlinger et al. [3] conducted a comprehensive survey analyzing open-source and commercial data quality tools, including Great Expectations, Deequ, MobyDQ, and Apache Griffin. The authors evaluated tools against five standard quality dimensions established in ISO 8000 and DAMA-DMBOK: completeness, accuracy, timeliness, consistency, and validity.

The survey revealed critical gaps in the existing tool landscape:
- Most existing tools focus strictly on static rule checking and fail to incorporate statistical drift monitoring or unsupervised machine learning.
- Many tools lack transparent, standardized scoring mechanisms, reporting disparate pass/fail lists rather than normalized quality indicators.
- User interface accessibility varies widely, with many enterprise tools requiring complex command-line orchestration or commercial cloud licensing.  
*Distinction in Our Work:* Our architecture addresses the gaps identified by Ehrlinger et al. by unifying rule validation, statistical drift testing, and machine learning anomaly detection into a single self-contained application with a unified composite score.

### 5.4 A Survey on Dataset Quality in Machine Learning (Budach et al., Information and Software Technology, 2024)
Budach et al. [4] provided a systematic taxonomy of dataset quality issues across the machine learning lifecycle, examining their practical effects on model accuracy, fairness, and robustness. The authors classified data defects into:
- **Low-level syntactic errors:** Missing cells, invalid formatting, duplicate records.
- **High-level semantic errors:** Inconsistent labels, multivariate outliers.
- **Contextual / Temporal errors:** Non-stationary covariate drift and target concept drift.

Budach et al. emphasized that detecting multivariate outliers is significantly more challenging than univariate filtering because anomalous observations often lie entirely within the marginal minimum and maximum boundaries of each individual feature, becoming detectable only when examining cross-feature correlations.  
*Distinction in Our Work:* Our system directly implements this insight by pairing univariate Pandera range validation with multivariate Isolation Forest modeling, ensuring both marginal and joint distribution defects are identified.

### 5.5 Auto-Validate by-History (Zhang et al., Microsoft Research / VLDB, 2023)
Zhang et al. [5] addressed the major operational bottleneck of rule-based data quality systems: the human burden of authoring and maintaining hundreds of declarative validation rules. The authors developed **Auto-Validate by-History (AVH)** for recurring data pipelines at Microsoft. AVH mines historical pipeline execution logs to automatically synthesize constraints and acceptable metric boundaries without human intervention.

Key findings from Zhang et al. include:
- Static constraints inevitably become obsolete as underlying business domains evolve, resulting in false positive alert fatigue.
- Tracking longitudinal metric drift over historical runs enables distinguishing between natural cyclical variations and genuine data corruption.  
*Distinction in Our Work:* While our current system uses a declarative JSON configuration (`config/validation_rules.json`) supplemented by a baseline reference dataset, the longitudinal monitoring principles established by Zhang et al. provide the theoretical foundation for our distribution drift testing module.

---

## 6. Existing System and Limitations

### 6.1 Description of Current Practices
In typical enterprise and academic data workflows, data validation is performed through one of three fragmented approaches:
1. **Ad-Hoc Manual Scripts:** Individual data scientists write standalone Jupyter notebook snippets using `df.isna().sum()`, `df.describe()`, or manual filtering.
2. **Standard Database Constraints:** Relational database management systems (RDBMS) enforce basic SQL constraints (`NOT NULL`, `UNIQUE`, `CHECK (age > 0)`).
3. **Heavyweight Enterprise MLOps Platforms:** Commercial cloud monitoring platforms (e.g., Monte Carlo, Datadog Data Observability) ingest pipeline metadata via cloud agents.

### 6.2 Limitations of the Existing Approaches
- **Lack of Integration:** Ad-hoc scripts lack reproducibility, automated reporting, and standardized scoring across teams.
- **Blindness to Multivariate Interactions:** SQL constraints evaluate each column independently and cannot detect multivariate anomalies where combinations of benign values represent operational failures.
- **Inability to Detect Covariate Shift:** Database constraints cannot measure distribution divergence ($P(X)$ drift) between training and production batches.
- **Cloud Dependency & Privacy Violations:** Commercial observability platforms require streaming sensitive customer data or database logs to third-party SaaS servers, creating compliance risks.
- **Scalability and Overhead:** Heavyweight enterprise tools require complex infrastructure (Kafka clusters, Spark engines, Kubernetes deployments) that are impractical for local auditing, exploratory data science, or academic project demonstrations.

---

## 7. Proposed System

The proposed **Data Quality Monitoring and Anomaly Detection System** provides a unified, lightweight, and 100% offline desktop/web solution. It combines deterministic constraint enforcement, non-parametric statistical hypothesis testing, and tree-based machine learning into an integrated pipeline.

### 7.1 Key Advantages of the Proposed Architecture
1. **Multi-Layered Defect Coverage:** Catches syntactic errors (nulls, duplicates), semantic violations (boundary breaches), distribution shifts (covariate drift), and multivariate outliers in a single execution run.
2. **Complete In-Memory Privacy:** Operates entirely within local process RAM. Zero data is persisted to disk or transmitted across network interfaces.
3. **Transparent & Explainable Scoring:** Provides a mathematically bounded composite Data Quality Score ($\text{DQS} \in [0, 100]$) with dynamic weight normalization, accompanied by explicit IQR-based feature deviation explanations for anomaly records.
4. **Interactive & Exportable Diagnostics:** Combines a reactive Streamlit web dashboard with instant, on-the-fly compilation of styled Excel workbooks and multi-page academic PDF reports.

---

## 8. Requirements and Technology Stack

### 8.1 Software & Hardware Requirements
- **Runtime Environment:** Python 3.11, 3.12, or 3.14.
- **Operating System:** Platform-agnostic (Windows 10/11, macOS, Linux).
- **Minimum Hardware:** Dual-core CPU, 4 GB RAM (8 GB recommended for large datasets).
- **Execution Mode:** Local offline execution; no internet connection required at runtime.

### 8.2 Technology Stack

| Component | Library & Version | Selection Rationale |
| :--- | :--- | :--- |
| **Data Ingestion & Manipulation** | Pandas `2.3.3`, NumPy `2.5.4` | Industry-standard vectorized tabular data manipulation and numerical array operations. |
| **Declarative Validation** | Pandera `0.35.0` | Vectorized DataFrame schema enforcement with exact row-index failure logging. |
| **Statistical Drift Testing** | SciPy `1.18.1`, Evidently `0.7.23` | Non-parametric statistical tests (Kolmogorov-Smirnov, Pearson Chi-Square, Wasserstein distance). |
| **Anomaly Detection** | Scikit-learn `1.9.1` | Optimized Isolation Forest implementation with sub-linear partition scaling. |
| **User Interface** | Streamlit `1.66.1`, Plotly `5.24.1` | Reactive web UI with interactive GPU-accelerated JavaScript charts. |
| **Excel Generation** | openpyxl `3.1.5` | Native Office Open XML generation with styled header bands, auto-filters, and freeze panes. |
| **PDF Generation** | ReportLab `5.1.0` | Flowable document layout via Platypus; dynamic two-pass page numbering via `NumberedCanvas`. |
| **Automated Testing** | Pytest `9.1.1` | Comprehensive automated test orchestration and edge-case verification. |

---

## 9. System Architecture

```
   +────────────────────────────────────────────────────────────────────────+
   |                       SYSTEM ARCHITECTURE PIPELINE                     |
   +────────────────────────────────────────────────────────────────────────+
                                       |
        +──────────────────────────────┴──────────────────────────────+
        |                                                             |
        v                                                             v
   [ Target Dataset ]                                       [ Reference Dataset ]
   (CSV / XLSX bytes)                                       (Optional Baseline)
        |                                                             |
        +──────────────────────────────┬──────────────────────────────+
                                       v
                    +─────────────────────────────────────+
                    | In-Memory Parsing & Defensive Copy  |
                    | (Data immutability guaranteed)      |
                    +──────────────────┬──────────────────+
                                       |
              +────────────────────────┴────────────────────────+
              v                                                 v
   +──────────────────────+                           +───────────────────+
   |  Syntactic Profiler  |                           |  Pandera Schema   |
   |  - Null cell counts  |                           |  - Range bounds   |
   |  - Duplicate rows    |                           |  - Non-null rules |
   |  - Memory footprint  |                           |  - Allowed sets   |
   +──────────┬───────────+                           +─────────┬─────────+
              |                                                 |
              +────────────────────────┬────────────────────────+
                                       v
                     +───────────────────────────────────+
                     | Composite Data Quality Scorecard  |
                     | DQS in [0, 100] (Normalized)      |
                     +─────────────────┬─────────────────+
                                       |
              +────────────────────────┴────────────────────────+
              v                                                 v
   +──────────────────────+                           +───────────────────+
   |  Statistical Drift   |                           | Isolation Forest  |
   |  - Two-Sample K-S    |                           | - Median impute   |
   |  - Wasserstein-1     |                           | - Path length     |
   |  - Pearson Chi-Square|                           | - Top IQR dev.    |
   +──────────┬───────────+                           +─────────┬─────────+
              |                                                 |
              +────────────────────────┬────────────────────────+
                                       v
                     +───────────────────────────────────+
                     |   Presentation & Export Center    |
                     | - Streamlit Reactive Dashboard    |
                     | - Styled openpyxl Workbook (.xlsx)|
                     | - ReportLab Platypus PDF (.pdf)   |
                     | - Raw Streaming CSV Logs          |
                     +───────────────────────────────────+
```

---

## 10. Methodology and Algorithms

### 10.1 Declarative Schema Validation Engine
Let dataset $\mathcal{D}$ comprise $N$ rows and $M$ columns. A schema contract $\mathcal{S}$ is defined as a collection of column specifications:

$$\mathcal{S} = \{ (c_j, \tau_j, \mathcal{C}_j) \}_{j=1}^M$$

where $c_j$ is the column name, $\tau_j$ is the expected datatype, and $\mathcal{C}_j = \{ \kappa_1, \kappa_2, \dots \}$ is a set of unary check predicates. Each check predicate $\kappa: x \to \{\text{True}, \text{False}\}$ asserts a business invariant:
- **Range Constraint:** $\kappa_{\text{range}}(x) \iff L_j \le x \le U_j$
- **Set Membership Constraint:** $\kappa_{\text{set}}(x) \iff x \in \Omega_j$
- **Non-Null Constraint:** $\kappa_{\text{not\_null}}(x) \iff x \notin \{\text{NaN}, \text{None}, \text{""}\}$

The validation engine evaluates all constraints vectorially across $\mathcal{D}$. Any evaluated cell where $\kappa(x) = \text{False}$ is logged as a structured `ValidationIssue` record containing:
$$\langle \text{IssueType}, \text{Column}, \text{RowIndex}, \text{Severity}, \text{ObservedValue}, \text{Description} \rangle$$

---

### 10.2 Statistical Distribution Drift Engine

```
   CUMULATIVE DISTRIBUTION (Two-Sample K-S)        ISOLATION TREE PARTITIONING
   
   1.0┌──────────────────────────────┐              Root [x1 < 45.2]
      │           /─── F_ref(x)      │             /                \
      │          /│                  │       Left child         Right child
      │         / │ D_max            │       [x2 < 0.12]        (Nominal cluster)
   0.5│        /  │                  │       /         \
      │       /   │                  │   [x_anom]    [x_norm]
      │      /────/─── F_cur(x)      │   (Isolated   (Requires more
   0.0└─────/────────────────────────┘    at depth 2) partitions)
      x_min                         x_max
```

#### Continuous Features: Two-Sample Kolmogorov-Smirnov Test
To determine whether a continuous production column $X_{\text{cur}}$ has drifted from its baseline reference distribution $X_{\text{ref}}$, the system computes the empirical cumulative distribution functions (eCDFs):

$$F_{\text{ref}, n}(x) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}_{(-\infty, x]}(X_{\text{ref}, i}), \quad F_{\text{cur}, m}(x) = \frac{1}{m} \sum_{j=1}^m \mathbb{I}_{(-\infty, x]}(X_{\text{cur}, j})$$

The test statistic $D_{n, m}$ represents the supremum absolute difference between the eCDFs:

$$D_{n, m} = \sup_{x \in \mathbb{R}} |F_{\text{ref}, n}(x) - F_{\text{cur}, m}(x)|$$

Under the null hypothesis $H_0: F_{\text{ref}} = F_{\text{cur}}$, the test statistic is evaluated against the asymptotic Kolmogorov distribution:

$$P\left( \sqrt{\frac{nm}{n+m}} D_{n, m} \le z \right) \to L(z) = 1 - 2 \sum_{k=1}^\infty (-1)^{k-1} e^{-2 k^2 z^2}$$

If the computed $p$-value satisfies $p < \alpha$ (default $\alpha = 0.05$), $H_0$ is rejected, and statistical distribution drift is formally flagged.

#### Physical Distance: Wasserstein-1 Distance
To measure the magnitude of drift independently of sample size $N$, the first Wasserstein distance (Earth Mover's Distance) is computed:

$$W_1(u, v) = \int_{-\infty}^\infty |F_{\text{ref}, n}(x) - F_{\text{cur}, m}(x)| \, dx$$

#### Categorical Features: Pearson's Chi-Square Test
For categorical variables with $C$ classes, observed frequencies in the current batch ($O_c$) are compared against expected frequencies ($E_c$) derived from reference baseline proportions ($p_{\text{ref}, c} = n_c / n$):

$$E_c = m \cdot p_{\text{ref}, c}$$

The test statistic follows a chi-square distribution with $C - 1$ degrees of freedom:

$$\chi^2 = \sum_{c=1}^C \frac{(O_c - E_c)^2}{E_c} \sim \chi^2_{C - 1}$$

If categories appear with zero frequency in the baseline ($E_c = 0$), the system falls back to the symmetric **Jensen-Shannon Divergence ($JSD$)**:

$$JSD(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)$$

---

### 10.3 Unsupervised Multivariate Anomaly Detection Engine

#### Isolation Tree Construction
Isolation Forest partitions instances using an ensemble of $T$ isolation trees ($T=100$). For each tree, a random subsample $\psi = 256$ is drawn. The tree recursively splits features at random values $p \in [\min(x_{\cdot, q}), \max(x_{\cdot, q})]$ until instances are isolated or maximum depth $h_{\max} = \lceil \log_2 \psi \rceil$ is reached.

#### Average Path Length of Unsuccessful Searches
Because isolation trees possess an equivalent structure to Binary Search Trees (BST), the average path length of unsuccessful searches over $\psi$ points provides the mathematical normalizing constant:

$$c(\psi) = 2 \left( \ln(\psi - 1) + \gamma \right) - \frac{2(\psi - 1)}{\psi}$$

where $\gamma \approx 0.5772156649$ is Euler's constant.

#### Anomaly Suspicion Score
For an observation $x$, let $h_t(x)$ denote the path length in tree $t$. The expected path length across the ensemble is $\mathbb{E}(h(x)) = \frac{1}{T} \sum_{t=1}^T h_t(x)$. The continuous suspicion score $s(x, \psi) \in [0, 1]$ is:

$$s(x, \psi) = 2^{-\frac{\mathbb{E}(h(x))}{c(\psi)}}$$

- When $\mathbb{E}(h(x)) \to 0 \implies s \to 1$: Observation requires few splits $\to$ **High Anomaly Probability**.
- When $\mathbb{E}(h(x)) \to c(\psi) \implies s \to 0.5$: Observation exhibits average path length $\to$ **Nominal Centrality**.
- When $\mathbb{E}(h(x)) \to \psi - 1 \implies s \to 0$: Observation requires maximum splits $\to$ **Deeply Inlying Nominal Point**.

#### Explainability: Normalized IQR Feature Deviation
For each observation flagged as anomalous, the system calculates the relative deviation from the baseline median in units of Interquartile Range ($IQR$):

$$\Delta_{i, j} = \frac{x_{i, j} - \text{Median}(X_j)}{Q_{75}(X_j) - Q_{25}(X_j)}$$

The features with the largest absolute deviation $|\Delta_{i, j}|$ are highlighted in the user interface and audit logs as the primary contributors to the observation's isolation.

---

## 11. Data Quality Score Formula

The composite **Data Quality Score ($\text{DQS}$)** is formulated as a transparent, weighted linear index:

$$\text{DQS} = \frac{\sum_{i \in \mathcal{A}} w_i S_i}{\sum_{i \in \mathcal{A}} w_i} \in [0, 100]$$

where $\mathcal{A} \subseteq \{\text{comp}, \text{uniq}, \text{val}\}$ denotes the set of active, applicable dimensions.

### 11.1 Dimension Formulas
1. **Completeness Score ($S_{\text{comp}}$, starting weight $w_{\text{comp}} = 40.0$):**
   $$S_{\text{comp}} = (1 - \rho_{\text{miss}}) \times 100, \quad \rho_{\text{miss}} = \frac{1}{N \cdot M} \sum_{i=1}^N \sum_{j=1}^M \mathbb{I}_{\text{null}}(x_{i, j})$$
2. **Uniqueness Score ($S_{\text{uniq}}$, starting weight $w_{\text{uniq}} = 25.0$):**
   $$S_{\text{uniq}} = \left( 1 - \frac{R_{\text{dup}}}{N} \right) \times 100 = \frac{U}{N} \times 100$$
   where $R_{\text{dup}}$ is the count of duplicate row tuples, and $U$ is the count of distinct row tuples.
3. **Validity Score ($S_{\text{val}}$, starting weight $w_{\text{val}} = 35.0$):**
   $$S_{\text{val}} = \max\left(0, \left( 1 - \frac{V_{\text{fail}}}{N_{\text{eval}}} \right) \times 100 \right)$$
   where $V_{\text{fail}}$ is the total count of Pandera check failures, and $N_{\text{eval}}$ is the total candidate cell evaluations across configured rules.

### 11.2 Dynamic Weight Normalization
If no validation rules are configured, validity is excluded ($\text{val} \notin \mathcal{A}$). The weights are dynamically normalized over active dimensions:

$$w'_k = \frac{w_k}{\sum_{j \in \mathcal{A}} w_j}, \quad \forall k \in \mathcal{A}$$

$$\text{DQS} = \sum_{k \in \mathcal{A}} w'_k \cdot S_k$$

### 11.3 Qualitative Grade Mapping
- **Excellent:** $\text{DQS} \ge 85.0$
- **Good:** $70.0 \le \text{DQS} < 85.0$
- **Fair:** $50.0 \le \text{DQS} < 70.0$
- **Poor:** $\text{DQS} < 50.0$

> **Academic Disclaimer:** DQS is an engineering heuristic indicator defined specifically for this project; it is not an official ISO or IEEE standard.

---

## 12. Implementation Details

The system is structured into cohesive, decoupled Python modules within `src/`:

- **`src/data_loader.py`:** Provides `read_csv_bytes`, `read_excel_bytes`, `list_excel_sheets`, and `load_dataset`. Uses `openpyxl` as the Excel engine. Enforces data immutability and raises typed exceptions (`EmptyDataError`, `MalformedDataError`, `WorksheetNotFoundError`).
- **`src/synthetic_generator.py`:** Generates deterministic synthetic benchmark datasets (`baseline.csv`, `current.csv`, `multi_sheet_demo.xlsx`) using fixed random seeds. Injects known nulls, duplicates, rule violations, distribution shifts, and 8 planted multivariate anomalies.
- **`src/profiling.py`:** Computes column-level and dataset-level profile metrics (null counts, empty strings, infinities, uniqueness, memory usage).
- **`src/validation.py`:** Builds Pandera schemas from `config/validation_rules.json` and evaluates DataFrames, capturing issue records with severity ratings (`CRITICAL`, `ERROR`, `WARNING`).
- **`src/quality_score.py`:** Implements the DQS formula with dynamic weight normalization, grade mapping, and academic disclaimers.
- **`src/drift_detection.py`:** Implements schema difference comparisons, continuous Two-Sample K-S tests, Wasserstein distance, categorical Pearson Chi-Square tests, and Jensen-Shannon divergence.
- **`src/anomaly_detection.py`:** Executes Scikit-learn's `IsolationForest` with median imputation fitted strictly on reference data, returning suspicion scores $s \in [0, 1]$ and IQR feature deviation explanations.
- **`src/report_generator.py`:** Implements multi-format exports (6-tab openpyxl Excel workbooks with navy styling and freeze panes; ReportLab Platypus PDFs with two-pass `NumberedCanvas` page numbering; streaming CSV logs).
- **`src/ui_charts.py`:** Plotly chart generators for the DQS gauge, dimensional radar chart, missingness bars, distribution shift overlays, and anomaly scatter projections.
- **`app.py`:** Streamlit application providing 5 responsive tabs, sidebar ingestion controls, hyperparameter adjustment, and in-memory export downloads.

---

## 13. Testing and Experimental Setup

### 13.1 Automated Testing Architecture (Pytest)
The test suite resides in `tests/` and comprises 91 unit and integration tests across 6 modules:
1. `test_data_loader.py` (31 tests): CSV parsing, multi-sheet XLSX extraction, empty files, malformed bytes, rejection of `.xls`.
2. `test_profiling_validation_scoring.py` (18 tests): Missingness counting, duplicate row detection, Pandera boundary checking, DQS dynamic weight normalization.
3. `test_drift_detection.py` (12 tests): Unchanged distributions ($p > 0.05$), shifted continuous distributions ($p < 0.05$), categorical frequency surges, constant columns.
4. `test_anomaly_detection.py` (11 tests): Isolation Forest scoring, median imputation, schema alignment, feature deviations.
5. `test_report_generator.py` (10 tests): CSV byte generation, openpyxl workbook structure and styles, ReportLab `%PDF-` binary signature and multi-page flow.
6. `test_app_integration.py` (9 tests): Streamlit pipeline execution with and without baseline data, Plotly figure creation, download serialization.

### 13.2 Experimental Evaluation Protocol
To evaluate the system empirically, benchmark tests must be executed comparing known injected defects against detected outputs.

---

## 14. Results and Discussion

> **Note on Experimental Results:**  
> In accordance with academic integrity guidelines, experimental results and benchmark metrics must reflect actual measured executions. The sections below provide the standardized experiment template and structure. Values should be populated by executing benchmark runs using the reproducible synthetic datasets.

### 14.1 Reproducible Experiment Table Template

Below is the standardized template for recording experimental benchmark evaluations across data quality dimensions:

| Experiment ID | Dataset & Size ($N \times M$) | Introduced Issue / Defect | Detection Method / Algorithm | Expected Result | Actual Result | Precision (%) | Recall (%) | F1-Score (%) | Processing Time (s) | Observations & Viva Talking Points |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | `current.csv` ($508 \times 8$) | 171 injected missing cells across 6 columns | Syntactic Profiler (`src/profiling.py`) | Detect exactly 171 missing cells | *[To be measured]* | *[Calculate]* | *[Calculate]* | *[Calculate]* | *[Measure]* | Tests zero-error cell missingness detection. |
| **EXP-02** | `current.csv` ($508 \times 8$) | 18 duplicate row tuples ($7.7\%$) | Uniqueness Profiler (`src/profiling.py`) | Detect exactly 18 duplicate rows | *[To be measured]* | *[Calculate]* | *[Calculate]* | *[Calculate]* | *[Measure]* | Verifies exact row-hash uniqueness identification. |
| **EXP-03** | `current.csv` ($508 \times 8$) | Negative ages (e.g. $-12$) & bound overflows ($190$) | Pandera Range Check (`min_value=18, max_value=100`) | Flag 100% of out-of-bound age cells | *[To be measured]* | *[Calculate]* | *[Calculate]* | *[Calculate]* | *[Measure]* | Evaluates deterministic semantic boundary enforcement. |
| **EXP-04** | `current.csv` ($508 \times 8$) | Invalid category strings (`Cryptocurrency`) | Pandera Category Check (`isin(allowed_set)`) | Flag all unlisted categorical values | *[To be measured]* | *[Calculate]* | *[Calculate]* | *[Calculate]* | *[Measure]* | Confirms categorical set membership validation. |
| **EXP-05** | Baseline ($500$) vs Current ($508$) | Continuous distribution shift on `transaction_amount` | Two-Sample Kolmogorov-Smirnov (`src/drift_detection.py`) | Reject $H_0$ ($p < 0.05$); flag drift | *[To be measured]* | N/A (Hypoth. test) | N/A | N/A | *[Measure]* | Validates sensitivity to scale and location shift. |
| **EXP-06** | Baseline ($500$) vs Current ($508$) | Stationary continuous features (`annual_income`) | Two-Sample Kolmogorov-Smirnov (`src/drift_detection.py`) | Accept $H_0$ ($p \ge 0.05$); no drift | *[To be measured]* | N/A (Hypoth. test) | N/A | N/A | *[Measure]* | Validates specificity (false positive alert control). |
| **EXP-07** | Baseline ($500$) vs Current ($508$) | Categorical surge in `Travel` category ($15\% \to 30\%$) | Pearson Chi-Square Test (`src/drift_detection.py`) | Reject $H_0$ ($p < 0.05$); flag drift | *[To be measured]* | N/A (Hypoth. test) | N/A | N/A | *[Measure]* | Confirms sensitivity to categorical frequency shifts. |
| **EXP-08** | `current.csv` ($508 \times 8$) | 8 planted multivariate anomalies | Isolation Forest (`contamination=0.05`) | All 8 planted records ranked in top outliers | *[To be measured]* | *[Calculate]* | *[Calculate]* | *[Calculate]* | *[Measure]* | Tests detection of complex cross-feature anomalies. |

### 14.2 Instructions for Populating Benchmark Experiments
1. Execute the automated benchmark script:
   ```powershell
   python -c "from src.synthetic_generator import create_demo_fixtures; create_demo_fixtures()"
   ```
2. Run the Streamlit application or test pipeline, and observe the output logs.
3. Compute Precision, Recall, and F1-Score for validation and anomaly detection using the ground-truth logs in `data/demo/DEMO_DATA_DOCUMENTATION.md`:
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
4. Record actual execution times in milliseconds/seconds to quantify computational overhead.

---

## 15. Limitations

While the proposed system provides an effective local monitoring solution, several inherent limitations must be documented:

1. **Large-Sample K-S Test Degeneracy:** As sample size $N \to \infty$, the standard error $\sqrt{(n+m)/(nm)}$ approaches zero. Consequently, trivial, practically imperceptible distribution variations produce $p < 0.05$. (The system mitigates this by reporting the Wasserstein-1 physical distance).
2. **Tabular Scope Restriction:** The architecture is designed specifically for structured 2D tabular formats (CSV and XLSX). Unstructured text documents, audio recordings, image tensors, and graph-structured data are outside the current processing scope.
3. **Median Imputation in Heavy Missingness:** When tabular datasets exhibit missingness rates exceeding $30-40\%$, median imputation prior to Isolation Forest modeling can artificially compress variance and distort cross-feature correlation structures.
4. **Batch Processing Orientation:** The system operates on discrete batches (files). It does not natively ingest infinite streaming event streams (e.g., Apache Kafka topics) without an external micro-batch windowing mechanism.

---

## 16. Conclusion and Future Scope

### 16.1 Conclusion
This project designed, implemented, and verified an end-to-end **Data Quality Monitoring and Anomaly Detection System**. By unifying deterministic schema validation (Pandera), non-parametric statistical hypothesis testing (Two-Sample K-S and Chi-Square tests), and unsupervised machine learning (Isolation Forest), the system overcomes the silent failure dilemma inherent in modern tabular data pipelines. Operating 100% in-memory with zero external network telemetry, the system provides an explainable Data Quality Score, responsive Plotly visual analytics, and multi-format audit reports (openpyxl XLSX, ReportLab PDF, CSV), verified across a 91-test automated Pytest suite.

### 16.2 Future Scope
1. **Streaming Data Integration:** Extend ingestion to support sliding-window micro-batch processing from Apache Kafka or Apache Pulsar streaming topics.
2. **Automated Constraint Synthesis:** Implement automated historical constraint mining (inspired by Microsoft's Auto-Validate by-History) to generate Pandera schema contracts from past clean batches.
3. **Active Learning & Human-in-the-Loop Feedback:** Allow domain experts to review flagged anomalies in the Streamlit interface and submit feedback, dynamically adjusting the Isolation Forest contamination decision boundary.
4. **Automated Data Remediation:** Develop rule-based auto-cleansing mechanisms (e.g., automated bounding, deduplication, imputation) with audit-trailed rollback capabilities.

---

## 17. References

1. **Breck, E., Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2019).** Data Validation for Machine Learning. *Proceedings of Machine Learning and Systems (MLSys / SysML 2019)*, 1, 334–347. URL: https://research.google/pubs/data-validation-for-machine-learning/
2. **Schelter, S., Lange, D., Schmidt, P., Celikel, M., Biessmann, F., & Grafberger, A. (2018).** Automating Large-Scale Data Quality Verification. *Proceedings of the VLDB Endowment (PVLDB)*, 11(12), 1781–1794. URL: https://www.vldb.org/pvldb/vol11/p1781-schelter.pdf
3. **Ehrlinger, L., Rechberger, M., & Wöß, W. (2022).** A Survey of Data Quality Measurement and Monitoring Tools. *Frontiers in Big Data*, 5, 850611. DOI: https://doi.org/10.3389/fdata.2022.850611
4. **Budach, L., Feuerpfeil, M., Ihde, N., Nakladal, A., Nejad, K., Rostami, S., & Bux, M. (2024).** A Survey on Dataset Quality in Machine Learning. *Information and Software Technology*, 166, 107268. DOI: https://doi.org/10.1016/j.infsof.2023.107268
5. **Zhang, C., Shen, Y., & Curino, C. (2023).** Auto-Validate by-History: Auto-Program Data Quality Constraints to Validate Recurring Data Pipelines. *Proceedings of the VLDB Endowment (PVLDB)*. Microsoft Research. URL: https://www.microsoft.com/en-us/research/publication/auto-validate-by-history-auto-program-data-quality-constraints-to-validate-recurring-data-pipelines-2/
6. **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).** Isolation Forest. *Eighth IEEE International Conference on Data Mining (ICDM 2008)*, 413–422.
7. **Kolmogorov, A. N. (1933).** Sulla determinazione empirica di una legge di distribuzione. *Giornale dell'Istituto Italiano degli Attuari*, 4, 83–91.
8. **Smirnov, N. (1948).** Table for Estimating the Goodness of Fit of Empirical Distributions. *The Annals of Mathematical Statistics*, 19(2), 279–281.
9. **Ramdas, A., Trillos, N. G., & Cuturi, M. (2017).** On Wasserstein Two-Sample Testing and Related Families of Nonparametric Tests. *Entropy*, 19(2), 47.
10. **International Organization for Standardization. (2015).** *ISO 8000: Data Quality*. Geneva, Switzerland.
