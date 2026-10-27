# Demo Benchmark Datasets: Documentation of Injected Defects

**Generated Files:**
- `baseline.csv` (1200 rows, 8 columns) - Clean reference dataset.
- `current.csv` (1226 rows, 8 columns) - Flawed and drifted production batch.
- `multi_sheet_demo.xlsx` - Multi-sheet workbook containing both 'Baseline' and 'Current'.

---

## 1. Deliberately Injected Defects in `current.csv`

### A. Missing Values (Completeness Defect)
- `annual_income`: 60 missing values (`np.nan`, 5.0%)
- `debt_to_income`: 48 missing values (`np.nan`, 4.0%)

### B. Duplicate Records (Uniqueness Defect)
- Injected: 18 exact duplicate rows (rows copied from indices 50..67).

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
8 multivariate outlier records injected (IDs `ANOM-00001` to `ANOM-00008`):
- **ANOM-00001**: Teenager with extreme executive income, terrible credit, massive transaction on basic account.
- **ANOM-00002**: Low-income retiree executing transaction 4x their annual income with near-1.0 DTI.
- **ANOM-00003**: Young business account with subprime credit and single transaction exceeding annual wage.
- **ANOM-00004**: Top 1% earner with severe default-level credit score and massive single charge.
- **ANOM-00005**: Average income earner with critical debt burden and enormous retail transaction.
- **ANOM-00006**: Near-perfect 840 credit score at age 20 with poverty-level income and luxury transactions.
- **ANOM-00007**: Six-figure utility bill payment even for a business account.
- **ANOM-00008**: Low-wage earner spending $56k on entertainment.
