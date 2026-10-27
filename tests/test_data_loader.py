"""Unit tests for src/data_loader.py and src/synthetic_generator.py.

Verifies:
- CSV loading from bytes with comma, semicolon, tab delimiters
- XLSX loading with openpyxl engine
- Worksheet listing and selection
- Validation of extensions (including legacy .xls rejection)
- Defensive error handling: 0 bytes, whitespace-only, header-only, malformed bytes
- DataFrame immutability guarantees
- Reproducibility and defect verification of synthetic generator
"""

import io
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

from src.data_loader import (
    DataLoaderError,
    EmptyDataError,
    MalformedDataError,
    UnsupportedFileFormatError,
    WorksheetNotFoundError,
    list_excel_sheets,
    load_dataset,
    read_csv_bytes,
    read_excel_bytes,
    validate_file_extension,
)
from src.synthetic_generator import (
    generate_baseline_dataset,
    generate_current_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv_bytes() -> bytes:
    """Fixture providing simple valid CSV bytes."""
    csv_text = "id,name,value\n1,Alice,100.5\n2,Bob,200.0\n3,Charlie,300.75\n"
    return csv_text.encode("utf-8")


@pytest.fixture
def sample_xlsx_bytes() -> bytes:
    """Fixture providing a valid 2-sheet in-memory Excel workbook."""
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Summary"
    ws1.append(["Metric", "Score"])
    ws1.append(["Quality", 95])
    ws1.append(["Drift", 0])

    ws2 = wb.create_sheet(title="RawData")
    ws2.append(["id", "reading"])
    ws2.append([101, 12.5])
    ws2.append([102, 14.8])

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# Extension Validation Tests
# ---------------------------------------------------------------------------

class TestExtensionValidation:
    """Tests for validate_file_extension."""

    def test_valid_csv_extensions(self):
        assert validate_file_extension("data.csv") == ".csv"
        assert validate_file_extension("DATA.CSV") == ".csv"
        assert validate_file_extension("path/to/my_data.csv") == ".csv"

    def test_valid_xlsx_extensions(self):
        assert validate_file_extension("workbook.xlsx") == ".xlsx"
        assert validate_file_extension("REPORT.XLSX") == ".xlsx"

    def test_reject_legacy_xls(self):
        with pytest.raises(UnsupportedFileFormatError) as exc_info:
            validate_file_extension("old_file.xls")
        assert "Legacy Excel (.xls, BIFF8) is not supported" in str(exc_info.value)

    def test_reject_unsupported_extensions(self):
        for bad_ext in ["data.txt", "data.json", "data.parquet", "data.pdf", "data.tsv"]:
            with pytest.raises(UnsupportedFileFormatError):
                validate_file_extension(bad_ext)

    def test_reject_empty_or_invalid_filename(self):
        with pytest.raises(UnsupportedFileFormatError):
            validate_file_extension("")
        with pytest.raises(UnsupportedFileFormatError):
            validate_file_extension(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CSV Ingestion Tests
# ---------------------------------------------------------------------------

class TestCSVLoading:
    """Tests for read_csv_bytes."""

    def test_read_standard_comma_csv(self, sample_csv_bytes):
        df = read_csv_bytes(sample_csv_bytes)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["id", "name", "value"]
        assert df.loc[0, "name"] == "Alice"
        assert df.loc[2, "value"] == 300.75

    def test_read_semicolon_delimited_csv(self):
        content = "col_a;col_b;col_c\n10;hello;1.1\n20;world;2.2\n".encode("utf-8")
        df = read_csv_bytes(content)
        assert len(df) == 2
        assert list(df.columns) == ["col_a", "col_b", "col_c"]
        assert df.loc[0, "col_b"] == "hello"

    def test_read_tab_delimited_csv(self):
        content = "x\ty\tz\n1\t2\t3\n4\t5\t6\n".encode("utf-8")
        df = read_csv_bytes(content)
        assert len(df) == 2
        assert list(df.columns) == ["x", "y", "z"]

    def test_read_latin1_encoded_csv(self):
        # String with Latin-1 accented character
        content = "id,city\n1,Montréal\n2,Zürich\n".encode("latin-1")
        df = read_csv_bytes(content)
        assert len(df) == 2
        assert df.loc[0, "city"] == "Montréal"

    def test_empty_zero_bytes_csv_raises(self):
        with pytest.raises(EmptyDataError) as exc_info:
            read_csv_bytes(b"")
        assert "empty (0 bytes)" in str(exc_info.value)

    def test_whitespace_only_csv_raises(self):
        with pytest.raises(EmptyDataError) as exc_info:
            read_csv_bytes(b"   \n\t  \r\n  ")
        assert "whitespace" in str(exc_info.value)

    def test_header_only_zero_rows_csv_raises(self):
        content = "col1,col2,col3\n".encode("utf-8")
        with pytest.raises(EmptyDataError) as exc_info:
            read_csv_bytes(content)
        assert "no data rows" in str(exc_info.value)

    def test_immutability_of_returned_dataframe(self, sample_csv_bytes):
        df1 = read_csv_bytes(sample_csv_bytes)
        df2 = read_csv_bytes(sample_csv_bytes)
        df1.loc[0, "name"] = "MUTATED"
        assert df2.loc[0, "name"] == "Alice"


# ---------------------------------------------------------------------------
# Excel (.xlsx) Ingestion Tests
# ---------------------------------------------------------------------------

class TestExcelLoading:
    """Tests for read_excel_bytes and list_excel_sheets."""

    def test_list_sheets(self, sample_xlsx_bytes):
        sheets = list_excel_sheets(sample_xlsx_bytes)
        assert sheets == ["Summary", "RawData"]

    def test_read_default_sheet(self, sample_xlsx_bytes):
        df = read_excel_bytes(sample_xlsx_bytes)
        assert list(df.columns) == ["Metric", "Score"]
        assert len(df) == 2
        assert df.loc[0, "Metric"] == "Quality"

    def test_read_named_sheet(self, sample_xlsx_bytes):
        df = read_excel_bytes(sample_xlsx_bytes, sheet_name="RawData")
        assert list(df.columns) == ["id", "reading"]
        assert len(df) == 2
        assert df.loc[0, "reading"] == 12.5

    def test_read_sheet_by_index(self, sample_xlsx_bytes):
        df = read_excel_bytes(sample_xlsx_bytes, sheet_name=1)
        assert list(df.columns) == ["id", "reading"]

    def test_nonexistent_sheet_name_raises(self, sample_xlsx_bytes):
        with pytest.raises(WorksheetNotFoundError) as exc_info:
            read_excel_bytes(sample_xlsx_bytes, sheet_name="DoesNotExist")
        assert "was not found" in str(exc_info.value)

    def test_out_of_range_sheet_index_raises(self, sample_xlsx_bytes):
        with pytest.raises(WorksheetNotFoundError):
            read_excel_bytes(sample_xlsx_bytes, sheet_name=99)

    def test_empty_zero_bytes_excel_raises(self):
        with pytest.raises(EmptyDataError):
            read_excel_bytes(b"")

    def test_corrupted_excel_bytes_raises(self):
        corrupted = b"PK\x03\x04definitely_not_a_valid_excel_archive_bytes_12345"
        with pytest.raises(MalformedDataError):
            read_excel_bytes(corrupted)

    def test_list_sheets_on_corrupted_bytes_raises(self):
        with pytest.raises(MalformedDataError):
            list_excel_sheets(b"Not An Excel File")


# ---------------------------------------------------------------------------
# Unified Dispatcher Tests
# ---------------------------------------------------------------------------

class TestLoadDatasetDispatcher:
    """Tests for load_dataset unified entrypoint."""

    def test_dispatch_csv(self, sample_csv_bytes):
        df = load_dataset(sample_csv_bytes, "customers.csv")
        assert len(df) == 3

    def test_dispatch_xlsx(self, sample_xlsx_bytes):
        df = load_dataset(sample_xlsx_bytes, "report.xlsx", sheet_name="RawData")
        assert len(df) == 2

    def test_dispatch_unsupported_raises(self, sample_csv_bytes):
        with pytest.raises(UnsupportedFileFormatError):
            load_dataset(sample_csv_bytes, "archive.zip")


# ---------------------------------------------------------------------------
# Synthetic Benchmark Generator Tests
# ---------------------------------------------------------------------------

class TestSyntheticGenerator:
    """Tests for reproducible synthetic demo dataset generation."""

    def test_baseline_dataset_reproducibility(self):
        df1 = generate_baseline_dataset(n_rows=200, seed=42)
        df2 = generate_baseline_dataset(n_rows=200, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_baseline_different_seeds_diverge(self):
        df1 = generate_baseline_dataset(n_rows=200, seed=42)
        df2 = generate_baseline_dataset(n_rows=200, seed=99)
        assert not df1["annual_income"].equals(df2["annual_income"])

    def test_baseline_schema_and_integrity(self):
        df = generate_baseline_dataset(n_rows=300, seed=42)
        expected_cols = [
            "customer_id",
            "age",
            "annual_income",
            "credit_score",
            "transaction_amount",
            "debt_to_income",
            "category",
            "account_type",
        ]
        assert list(df.columns) == expected_cols
        # Baseline should have ZERO missing values
        assert df.isna().sum().sum() == 0
        # Age should be within [18, 75]
        assert df["age"].min() >= 18
        assert df["age"].max() <= 75
        # Credit score within [350, 850]
        assert df["credit_score"].min() >= 350
        assert df["credit_score"].max() <= 850

    def test_current_dataset_reproducibility(self):
        df1, issues1 = generate_current_dataset(n_rows=200, seed=101)
        df2, issues2 = generate_current_dataset(n_rows=200, seed=101)
        pd.testing.assert_frame_equal(df1, df2)
        assert issues1 == issues2

    def test_current_dataset_injected_faults(self):
        df, issues = generate_current_dataset(n_rows=1200, seed=101)

        # 1. Verify missing values
        assert df["annual_income"].isna().sum() == 60
        assert df["debt_to_income"].isna().sum() == 48

        # 2. Verify duplicates
        assert df.duplicated().sum() >= 18

        # 3. Verify out-of-bounds rule violations
        # Invalid ages (<18 or >100)
        invalid_ages = df[~df["age"].between(18, 100)]
        assert len(invalid_ages) >= 4

        # Negative incomes
        negative_incomes = df[df["annual_income"] < 0]
        assert len(negative_incomes) == 5

        # Invalid credit scores (<300 or >850)
        invalid_credits = df[~df["credit_score"].between(300, 850)]
        assert len(invalid_credits) >= 4

        # 4. Verify presence of multivariate anomaly records
        anom_rows = df[df["customer_id"].str.startswith("ANOM-")]
        assert len(anom_rows) == 8

    def test_fixture_files_can_be_loaded_via_data_loader(self):
        demo_dir = Path("data/demo")
        baseline_path = demo_dir / "baseline.csv"
        current_path = demo_dir / "current.csv"
        excel_path = demo_dir / "multi_sheet_demo.xlsx"

        assert baseline_path.exists()
        assert current_path.exists()
        assert excel_path.exists()

        # Load CSVs from raw bytes
        baseline_df = load_dataset(baseline_path.read_bytes(), "baseline.csv")
        assert len(baseline_df) == 1200

        current_df = load_dataset(current_path.read_bytes(), "current.csv")
        assert len(current_df) > 1200

        # Load Excel sheets
        excel_bytes = excel_path.read_bytes()
        sheets = list_excel_sheets(excel_bytes)
        assert "Baseline" in sheets
        assert "Current" in sheets

        excel_baseline = load_dataset(excel_bytes, "demo.xlsx", sheet_name="Baseline")
        assert len(excel_baseline) == 1200
        excel_current = load_dataset(excel_bytes, "demo.xlsx", sheet_name="Current")
        assert len(excel_current) > 1200
