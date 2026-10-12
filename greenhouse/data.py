"""Workbook loading and normalization for greenhouse light-response data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


VARIABLE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("F", "F"),
    ("Fm′", "Fm_light"),
    ("P700ox", "P700ox"),
    ("Pm′", "Pm_light"),
    ("Y(II)", "Y(II)_calc"),
    ("ETR(II)", "ETR(II)_calc"),
    ("Fo′", "Fo_light_calc"),
    ("Y(NPQ)", "Y(NPQ)_calc"),
    ("Y(NO)", "Y(NO)_calc"),
    ("NPQ", "NPQ_calc"),
    ("qP", "qP_calc"),
    ("qL", "qL_calc"),
    ("qN", "qN_calc"),
    ("Y(ND)", "Y(ND)_calc"),
    ("Y(I)", "Y(I)_calc"),
    ("ETR(I)", "ETR(I)_calc"),
    ("Y(NA)", "Y(NA)_calc"),
)

VARIABLE_LABELS = {column: label for label, column in VARIABLE_OPTIONS}
REQUIRED_COLUMNS = {"Date", "Label", "Water", "Nitrogen", "PAR"}


class DataValidationError(ValueError):
    """Raised when a workbook cannot be normalized for plotting."""


def file_signature(path: Path) -> tuple[int, int]:
    """Return the values used to invalidate Streamlit's workbook cache."""

    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False)
def load_workbook(
    path_string: str,
    modified_time_ns: int,
    file_size: int,
) -> pd.DataFrame:
    """Read and normalize one crop workbook.

    The signature arguments are intentionally part of the cache key. They make a
    background file replacement visible without requiring a manual cache clear.
    """

    del modified_time_ns, file_size
    path = Path(path_string)
    try:
        frame = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:  # pandas/openpyxl provide useful details in the cause
        raise DataValidationError(f"Could not read {path.name}: {exc}") from exc

    normalized_names = [str(column).strip() for column in frame.columns]
    if len(normalized_names) != len(set(normalized_names)):
        raise DataValidationError(
            f"{path.name} contains duplicate column names after trimming spaces."
        )
    frame.columns = normalized_names

    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    identity_columns = [column for column in ("Block", "Replicate") if column in frame]
    if missing or not identity_columns:
        details: list[str] = []
        if missing:
            details.append(f"missing required columns: {', '.join(missing)}")
        if not identity_columns:
            details.append("missing Block or Replicate")
        raise DataValidationError(f"{path.name}: {'; '.join(details)}.")

    identity_column = identity_columns[0]
    parsed_dates = pd.to_datetime(frame["Date"], errors="coerce", format="mixed")
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_dates:
        raise DataValidationError(
            f"{path.name}: {invalid_dates:,} row(s) have an invalid Date. "
            "Use native Excel dates or MM-DD-YYYY text."
        )
    frame["Date"] = parsed_dates.dt.normalize()

    for column in ("Label", "Water", "Nitrogen"):
        values = frame[column].astype("string").str.strip()
        values = values.mask(values.eq(""))
        invalid = int(values.isna().sum())
        if invalid:
            raise DataValidationError(
                f"{path.name}: {invalid:,} row(s) have an empty {column}."
            )
        frame[column] = values.astype(str)

    numeric_par = pd.to_numeric(frame["PAR"], errors="coerce")
    valid_par = numeric_par.notna() & np.isfinite(numeric_par)
    if not bool(valid_par.all()):
        invalid = int((~valid_par).sum())
        raise DataValidationError(
            f"{path.name}: {invalid:,} row(s) have a non-numeric or infinite PAR."
        )
    frame["PAR"] = numeric_par.astype(float)

    identity_values = frame[identity_column].astype("string").str.strip()
    identity_values = identity_values.mask(identity_values.eq(""))
    if identity_values.isna().any():
        invalid = int(identity_values.isna().sum())
        raise DataValidationError(
            f"{path.name}: {invalid:,} row(s) have an empty {identity_column}."
        )
    frame["PlantUnit"] = identity_values.astype(str)
    frame["PlantUnitColumn"] = identity_column

    if "DAD" in frame:
        frame["DAD"] = pd.to_numeric(frame["DAD"], errors="coerce")

    for _, column in VARIABLE_OPTIONS:
        if column in frame:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = numeric.mask(~np.isfinite(numeric))

    frame["DateText"] = frame["Date"].dt.strftime("%Y-%m-%d")
    frame["SourceRow"] = np.arange(2, len(frame) + 2, dtype=int)
    return frame


def available_variables(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Return configured variables that contain at least one valid value."""

    available: list[tuple[str, str]] = []
    for label, column in VARIABLE_OPTIONS:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if bool((numeric.notna() & np.isfinite(numeric)).any()):
            available.append((label, column))
    return available


def valid_value_mask(frame: pd.DataFrame, variable: str) -> pd.Series:
    """Identify finite values for one response variable."""

    numeric = pd.to_numeric(frame[variable], errors="coerce")
    return numeric.notna() & np.isfinite(numeric)


def ordered_values(values: pd.Series, preferred: tuple[str, ...] = ()) -> list[str]:
    """Return stable category options, respecting a preferred scientific order."""

    unique = {str(value) for value in values.dropna().unique()}
    preferred_values = [value for value in preferred if value in unique]
    return preferred_values + sorted(unique.difference(preferred_values), key=str.casefold)


def format_date(date: pd.Timestamp, frame: pd.DataFrame) -> str:
    """Format a date and append DAD when it is uniquely defined for that date."""

    date = pd.Timestamp(date).normalize()
    label = date.strftime("%Y-%m-%d")
    if "DAD" not in frame:
        return label
    dads = frame.loc[frame["Date"].eq(date), "DAD"].dropna().unique()
    if len(dads) != 1:
        return label
    dad = float(dads[0])
    dad_text = str(int(dad)) if dad.is_integer() else f"{dad:g}"
    return f"{label} · DAD {dad_text}"

