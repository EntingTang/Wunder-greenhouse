"""Reusable plant-level and treatment-level aggregation rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from greenhouse.data import valid_value_mask


PLANT_GROUPS = ["Date", "Label", "Water", "Nitrogen", "PAR"]
TREATMENT_GROUPS = ["Date", "Water", "Nitrogen", "PAR"]
PERIOD_PLANT_GROUPS = ["Label", "Water", "Nitrogen", "PAR"]
PERIOD_TREATMENT_GROUPS = ["Water", "Nitrogen", "PAR"]


def _plant_metadata(frame: pd.DataFrame, include_date: bool) -> dict[str, tuple[str, str]]:
    aggregations: dict[str, tuple[str, str]] = {
        "PlantUnit": ("PlantUnit", "first"),
        "PlantUnitColumn": ("PlantUnitColumn", "first"),
        "RawRecords": ("SourceRow", "count"),
    }
    if include_date and "DAD" in frame:
        aggregations["DAD"] = ("DAD", "first")
    return aggregations


@st.cache_data(show_spinner=False)
def aggregate_individual(
    frame: pd.DataFrame,
    variable: str,
) -> tuple[pd.DataFrame, int, int]:
    """Average duplicate light-curve records for each plant, date, and PAR."""

    total = len(frame)
    valid_mask = valid_value_mask(frame, variable)
    invalid_count = int((~valid_mask).sum())
    valid = frame.loc[valid_mask].copy()
    if valid.empty:
        return pd.DataFrame(), invalid_count, total

    aggregations = {"Value": (variable, "mean"), **_plant_metadata(valid, True)}
    result = (
        valid.groupby(PLANT_GROUPS, observed=True, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["Date", "Label", "PAR"], kind="stable")
        .reset_index(drop=True)
    )
    result["DateText"] = result["Date"].dt.strftime("%Y-%m-%d")
    return result, invalid_count, total


@st.cache_data(show_spinner=False)
def aggregate_daily_treatment(
    frame: pd.DataFrame,
    variable: str,
) -> tuple[pd.DataFrame, int, int]:
    """Compute daily treatment mean, sample SD, and effective plant count."""

    plants, invalid_count, total = aggregate_individual(frame, variable)
    if plants.empty:
        return pd.DataFrame(), invalid_count, total

    result = (
        plants.groupby(TREATMENT_GROUPS, observed=True, dropna=False)
        .agg(
            Mean=("Value", "mean"),
            SD=("Value", lambda values: values.std(ddof=1)),
            n=("Value", "count"),
        )
        .reset_index()
        .sort_values(["Date", "Water", "Nitrogen", "PAR"], kind="stable")
        .reset_index(drop=True)
    )
    result["DateText"] = result["Date"].dt.strftime("%Y-%m-%d")
    if "DAD" in plants:
        dad_by_date = plants.groupby("Date", observed=True)["DAD"].first()
        result["DAD"] = result["Date"].map(dad_by_date)
    return result, invalid_count, total


@st.cache_data(show_spinner=False)
def aggregate_period_treatment(
    frame: pd.DataFrame,
    variable: str,
) -> tuple[pd.DataFrame, int, int]:
    """Apply the Facet-script period calculation in its original order.

    Records are averaged directly within plant and treatment across the selected
    dates. Dates with more valid records therefore contribute more weight to that
    plant's period mean.
    """

    total = len(frame)
    valid_mask = valid_value_mask(frame, variable)
    invalid_count = int((~valid_mask).sum())
    valid = frame.loc[valid_mask].copy()
    if valid.empty:
        return pd.DataFrame(), invalid_count, total

    plant_means = (
        valid.groupby(PERIOD_PLANT_GROUPS, observed=True, dropna=False)
        .agg(
            Value=(variable, "mean"),
            PlantUnit=("PlantUnit", "first"),
            PlantUnitColumn=("PlantUnitColumn", "first"),
            RawRecords=("SourceRow", "count"),
        )
        .reset_index()
    )
    result = (
        plant_means.groupby(PERIOD_TREATMENT_GROUPS, observed=True, dropna=False)
        .agg(
            Mean=("Value", "mean"),
            SD=("Value", lambda values: values.std(ddof=1)),
            n=("Value", "count"),
        )
        .reset_index()
        .sort_values(["Water", "Nitrogen", "PAR"], kind="stable")
        .reset_index(drop=True)
    )
    return result, invalid_count, total
