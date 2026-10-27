"""Data Loading & Safe Ingestion Module.

This module provides offline-first, memory-safe data ingestion utilities for
tabular datasets (.csv and .xlsx).

Design Principles:
- Immutability: Uploaded datasets are read in-memory and returned as deep copies.
- Zero Disk Writes: Files are parsed directly from bytes/memory streams.
- Explicit Error Handling: Custom typed exceptions for empty, malformed,
  or unsupported formats.
- Modern Standards: Enforces .xlsx (OpenXML) with openpyxl; explicitly rejects
  legacy .xls files.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Union

import openpyxl
import pandas as pd


# ---------------------------------------------------------------------------
# Custom Domain Exceptions
# ---------------------------------------------------------------------------

class DataLoaderError(Exception):
    """Base exception for all data loading errors."""
    pass


class UnsupportedFileFormatError(DataLoaderError):
    """Raised when an uploaded file has an unsupported format or extension."""
    pass


class EmptyDataError(DataLoaderError):
    """Raised when an uploaded file is empty (0 bytes) or lacks data rows/columns."""
    pass


class MalformedDataError(DataLoaderError):
    """Raised when an uploaded file cannot be parsed or decoded."""
    pass


class WorksheetNotFoundError(DataLoaderError):
    """Raised when the requested worksheet does not exist in the Excel workbook."""
    pass


# ---------------------------------------------------------------------------
# Format Validation
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def validate_file_extension(filename: str) -> str:
    """Validate that the file extension is supported.

    Args:
        filename: Name of the uploaded file.

    Returns:
        The normalized lowercase extension (e.g. '.csv', '.xlsx').

    Raises:
        UnsupportedFileFormatError: If the extension is not supported or is legacy .xls.
    """
    if not filename or not isinstance(filename, str):
        raise UnsupportedFileFormatError("Filename must be a non-empty string.")

    ext = Path(filename).suffix.lower()

    if ext == ".xls":
        raise UnsupportedFileFormatError(
            "Legacy Excel (.xls, BIFF8) is not supported due to security and "
            "compatibility constraints. Please convert or save your workbook as "
            "modern Excel (.xlsx, OpenXML) or CSV."
        )

    if ext not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileFormatError(
            f"Unsupported file extension '{ext}'. Supported formats are: {supported_str}."
        )

    return ext


# ---------------------------------------------------------------------------
# Excel Workbook Utilities
# ---------------------------------------------------------------------------

def list_excel_sheets(file_source: Union[bytes, io.BytesIO]) -> List[str]:
    """Inspect and list all worksheet names in an Excel (.xlsx) workbook.

    Uses openpyxl in read-only mode to extract sheet metadata without loading
    full worksheet cell contents into memory.

    Args:
        file_source: Raw file bytes or in-memory BytesIO buffer.

    Returns:
        List of worksheet names in workbook order.

    Raises:
        EmptyDataError: If the input buffer is empty.
        MalformedDataError: If the workbook is corrupted or cannot be parsed.
    """
    buffer = _ensure_bytes_io(file_source)

    try:
        workbook = openpyxl.load_workbook(buffer, read_only=True, keep_links=False)
        sheet_names = list(workbook.sheetnames)
        workbook.close()
    except openpyxl.utils.exceptions.InvalidFileException as exc:
        raise MalformedDataError(f"The provided file is not a valid Excel workbook: {exc}") from exc
    except Exception as exc:
        raise MalformedDataError(f"Failed to read Excel workbook structure: {exc}") from exc

    if not sheet_names:
        raise EmptyDataError("The Excel workbook contains no worksheets.")

    return sheet_names


# ---------------------------------------------------------------------------
# CSV Ingestion
# ---------------------------------------------------------------------------

def read_csv_bytes(
    file_source: Union[bytes, io.BytesIO],
    delimiter: str | None = None,
    encoding: str | None = None,
    filename: str | None = None,
) -> pd.DataFrame:
    """Read a CSV file from in-memory bytes into a Pandas DataFrame.

    Tries standard encodings (UTF-8, Latin-1 fallback) and handles common delimiters.
    Returns an immutable deep copy of the parsed data.

    Args:
        file_source: Raw file bytes or BytesIO stream.
        delimiter: Optional explicit delimiter. If None, auto-detected or defaults to comma.
        encoding: Optional explicit encoding. If None, attempts UTF-8 then Latin-1.

    Returns:
        pd.DataFrame: Deep copy of parsed tabular data.

    Raises:
        EmptyDataError: If the file is 0 bytes, contains only whitespace, or has no rows.
        MalformedDataError: If parsing fails due to malformed rows or encoding errors.
    """
    raw_bytes = _ensure_bytes(file_source)

    if len(raw_bytes) == 0:
        raise EmptyDataError("The uploaded CSV file is empty (0 bytes).")

    # Check for whitespace-only content
    if not raw_bytes.strip():
        raise EmptyDataError("The uploaded CSV file contains only whitespace.")

    # Encoding detection fallback
    encodings_to_try = [encoding] if encoding else ["utf-8", "latin-1", "cp1252"]
    last_error: Exception | None = None
    decoded_text: str | None = None

    for enc in encodings_to_try:
        try:
            decoded_text = raw_bytes.decode(enc)
            break
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
            continue

    if decoded_text is None:
        raise MalformedDataError(
            f"Failed to decode CSV bytes with supported encodings {encodings_to_try}: {last_error}"
        )

    # Delimiter resolution
    delimiters_to_try = [delimiter] if delimiter else [",", ";", "\t", "|"]
    df: pd.DataFrame | None = None

    for delim in delimiters_to_try:
        try:
            temp_df = pd.read_csv(
                io.StringIO(decoded_text),
                sep=delim,
                engine="python",
                on_bad_lines="error",
            )
            # A valid parse should have at least 1 column and not lump everything into a single column
            # if multiple delimiters were possible
            if delimiter or temp_df.shape[1] > 1 or len(delimiters_to_try) == 1:
                df = temp_df
                break
            if df is None:
                df = temp_df
        except pd.errors.EmptyDataError:
            raise EmptyDataError("The CSV file contains no data or columns.")
        except Exception as exc:
            last_error = exc
            continue

    if df is None:
        raise MalformedDataError(f"Failed to parse CSV file: {last_error}")

    return _validate_and_copy_dataframe(df)


# ---------------------------------------------------------------------------
# XLSX Ingestion
# ---------------------------------------------------------------------------

def read_excel_bytes(
    file_source: Union[bytes, io.BytesIO],
    sheet_name: str | int | None = None,
    filename: str | None = None,
) -> pd.DataFrame:
    """Read an Excel (.xlsx) file from in-memory bytes using openpyxl engine.

    Args:
        file_source: Raw file bytes or BytesIO stream.
        sheet_name: Specific worksheet name (str) or index (int). If None, loads the first sheet.

    Returns:
        pd.DataFrame: Deep copy of parsed worksheet data.

    Raises:
        EmptyDataError: If workbook is 0 bytes or worksheet is empty.
        WorksheetNotFoundError: If requested sheet name is not found.
        MalformedDataError: If openpyxl cannot parse the workbook.
    """
    raw_bytes = _ensure_bytes(file_source)

    if len(raw_bytes) == 0:
        raise EmptyDataError("The uploaded Excel file is empty (0 bytes).")

    # Verify sheet availability first
    available_sheets = list_excel_sheets(raw_bytes)

    target_sheet: str | int
    if sheet_name is None:
        target_sheet = available_sheets[0]
    elif isinstance(sheet_name, str):
        if sheet_name not in available_sheets:
            raise WorksheetNotFoundError(
                f"Worksheet '{sheet_name}' was not found. Available sheets: {available_sheets}"
            )
        target_sheet = sheet_name
    elif isinstance(sheet_name, int):
        if sheet_name < 0 or sheet_name >= len(available_sheets):
            raise WorksheetNotFoundError(
                f"Worksheet index {sheet_name} out of range [0, {len(available_sheets) - 1}]."
            )
        target_sheet = sheet_name
    else:
        raise WorksheetNotFoundError(f"Invalid sheet identifier: {sheet_name}")

    buffer = io.BytesIO(raw_bytes)
    try:
        df = pd.read_excel(
            buffer,
            sheet_name=target_sheet,
            engine="openpyxl",
        )
    except Exception as exc:
        raise MalformedDataError(f"Failed to parse Excel worksheet '{target_sheet}': {exc}") from exc

    return _validate_and_copy_dataframe(df)


# ---------------------------------------------------------------------------
# Unified Ingestion Entrypoint
# ---------------------------------------------------------------------------

def load_dataset(
    file_source: Union[bytes, io.BytesIO],
    filename: str,
    sheet_name: str | int | None = None,
) -> pd.DataFrame:
    """Load a tabular dataset (.csv or .xlsx) safely into an immutable DataFrame.

    Args:
        file_source: Raw file bytes or BytesIO buffer.
        filename: Original file name (used for format detection).
        sheet_name: Optional worksheet name/index if file is an Excel workbook.

    Returns:
        pd.DataFrame: Deep copy of loaded dataset.

    Raises:
        UnsupportedFileFormatError: If file extension is unsupported or legacy .xls.
        EmptyDataError: If file has 0 bytes or no usable data.
        MalformedDataError: If file is corrupted.
        WorksheetNotFoundError: If requested Excel sheet does not exist.
    """
    ext = validate_file_extension(filename)

    if ext == ".csv":
        return read_csv_bytes(file_source)
    elif ext == ".xlsx":
        return read_excel_bytes(file_source, sheet_name=sheet_name)
    else:
        raise UnsupportedFileFormatError(f"Unhandled file extension: {ext}")


# ---------------------------------------------------------------------------
# Internal Helper Functions
# ---------------------------------------------------------------------------

def _ensure_bytes(source: Union[bytes, io.BytesIO]) -> bytes:
    """Normalize file input to bytes."""
    if isinstance(source, bytes):
        return source
    elif isinstance(source, io.BytesIO):
        source.seek(0)
        return source.read()
    elif hasattr(source, "read"):
        source.seek(0)
        data = source.read()
        return data if isinstance(data, bytes) else data.encode("utf-8")
    raise TypeError(f"Expected bytes or BytesIO stream, got {type(source).__name__}")


def _ensure_bytes_io(source: Union[bytes, io.BytesIO]) -> io.BytesIO:
    """Normalize file input to a seekable BytesIO buffer."""
    if isinstance(source, io.BytesIO):
        source.seek(0)
        return source
    elif isinstance(source, bytes):
        return io.BytesIO(source)
    elif hasattr(source, "read"):
        source.seek(0)
        data = source.read()
        b = data if isinstance(data, bytes) else data.encode("utf-8")
        return io.BytesIO(b)
    raise TypeError(f"Expected bytes or BytesIO stream, got {type(source).__name__}")


def _validate_and_copy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate DataFrame dimensions and return a defensive deep copy."""
    if df.empty and df.shape[1] == 0:
        raise EmptyDataError("The dataset contains no usable rows or columns.")

    if df.shape[1] == 0:
        raise EmptyDataError("The dataset contains rows but zero usable columns.")

    if len(df) == 0:
        raise EmptyDataError("The dataset contains a header but no data rows (0 rows).")

    # Defensive copy guaranteeing caller immutability
    return df.copy(deep=True)
