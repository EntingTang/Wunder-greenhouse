"""Measured light responses and workbook-provided YeLP model validation."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from greenhouse.data import (
    DataValidationError,
    MODEL_NUMERIC_COLUMNS,
    file_signature,
    format_date,
    load_workbook,
    ordered_values,
    valid_value_mask,
)
from greenhouse.plotting import PlotSpec, axis_label
from greenhouse.ui import (
    NITROGEN_ORDER,
    WATER_ORDER,
    _date_controls,
    _date_summary,
    _number,
    _render_chart,
    _render_downloads,
    _single_choice,
    _slug,
    _state_key,
)


MODEL_VARIABLES = {
    "ETRII": ("ETR(II)", "ETR_fit_Ye", "Nk_N0_ETR", "etr_fit_success", "etr_fit_message"),
    "YII": ("Y(II)", "YII_fit_Ye", "Nk_N0_YII", "yii_ye_fit_success", "yii_ye_fit_message"),
}
LEAF_AREA_CM2 = 0.28274
PAR_UNIT = "µmol m⁻² s⁻¹"
INVERSE_PAR_UNIT = "m² s µmol⁻¹"
# Definitions follow Ye et al.'s mechanistic light-response model:
# https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1478346/full
COMMON_PARAMETERS = (
    ("PAR_sat", "PARₛₐₜ", PAR_UNIT, "Saturation light intensity"),
)
ETR_PARAMETERS = (
    ("alpha_e", "αₑ", "µmol e⁻ (µmol photons)⁻¹", "Initial slope"),
    ("beta_e", "βₑ", INVERSE_PAR_UNIT, "Photoinhibition coefficient"),
    ("gamma_e", "γₑ", INVERSE_PAR_UNIT, "Saturation coefficient"),
    ("gi_gk_ETR", "gᵢ/gₖ", "", "Ratio of ground-state to excited-state energy level degeneracies"),
    *COMMON_PARAMETERS,
    ("ETR_max", "Jₘₐₓ", "µmol e⁻ m⁻² s⁻¹", "Maximum electron transport rate"),
    ("sigma_ik", "σᵢₖ", "m²", "Intrinsic light absorption cross section"),
    ("tau_min", "τₘᵢₙ", "s", "Minimum average lifetime of pigment molecules in the excited state"),
    ("N0", "N₀", "", "Number of light harvesting pigments"),
    ("ETR_R2", "R²", "", "Determination coefficient"),
    ("ETR_rmse", "RMSE", "µmol e⁻ m⁻² s⁻¹", "Root mean squared error"),
)
YII_PARAMETERS = (
    ("beta_e_YII", "βₑ", INVERSE_PAR_UNIT, "Photoinhibition coefficient"),
    ("gamma_e_YII", "γₑ", INVERSE_PAR_UNIT, "Saturation coefficient"),
    ("gi_gk_YII", "gᵢ/gₖ", "", "Ratio of ground-state to excited-state energy level degeneracies"),
    *COMMON_PARAMETERS,
    ("YIImax_Ye", "Y(II)ₘₐₓ", "", "Maximum effective quantum yield of photosystem II"),
    ("YII_Ye_R2", "R²", "", "Determination coefficient"),
    ("YII_Ye_rmse", "RMSE", "", "Root mean squared error"),
)


@st.cache_data(show_spinner=False)
def load_model_workbook(
    path_string: str, modified_time_ns: int, file_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both sheets and validate the Label + Date parameter lookup."""

    curves = load_workbook(
        path_string, modified_time_ns, file_size, sheet_name="Curve_data"
    )
    path = Path(path_string)
    try:
        fits = pd.read_excel(path, sheet_name="fit_results", engine="openpyxl")
    except Exception as exc:
        raise DataValidationError(f"Could not read {path.name} fit_results: {exc}") from exc
    fits.columns = [str(column).strip() for column in fits.columns]
    if fits.columns.duplicated().any():
        raise DataValidationError(f"{path.name}: duplicate fit_results column names.")
    missing_curves = sorted({"DAD", "Block", *MODEL_NUMERIC_COLUMNS}.difference(curves.columns))
    required_fits = {
        "Date", "Label", "Chl", "etr_fit_success", "etr_fit_message",
        "yii_ye_fit_success", "yii_ye_fit_message",
        *(item[0] for item in ETR_PARAMETERS + YII_PARAMETERS),
    }
    missing_fits = sorted(required_fits.difference(fits.columns))
    if missing_curves or missing_fits:
        raise DataValidationError(
            f"{path.name}: missing Curve_data columns {missing_curves} "
            f"or fit_results columns {missing_fits}."
        )
    fits["Date"] = pd.to_datetime(fits["Date"], errors="coerce", format="mixed").dt.normalize()
    fits["Label"] = fits["Label"].astype("string").str.strip().replace("", pd.NA)
    if fits[["Date", "Label"]].isna().any().any():
        raise DataValidationError(f"{path.name}: fit_results has an invalid Date or Label.")
    if fits.duplicated(["Date", "Label"]).any():
        raise DataValidationError(f"{path.name}: fit_results has more than one fit for a Label + Date.")
    return curves, fits


def _parameter_table(row: pd.Series, parameters: tuple) -> pd.DataFrame:
    records = []
    for column, symbol, unit, description in parameters:
        value = _number(row.get(column))
        if unit:
            value += f" ({unit})"
        records.append({"Parameter": symbol, "Value": value, "Description": description})
    return pd.DataFrame(records)


def _render_parameters(
    fits: pd.DataFrame, dates: list[pd.Timestamp], label: str, variable: str, frame: pd.DataFrame
) -> None:
    st.markdown("#### Parameters")
    parameters = ETR_PARAMETERS if variable == "ETRII" else YII_PARAMETERS
    _, _, _, success_column, message_column = MODEL_VARIABLES[variable]
    for date in sorted(dates):
        st.markdown(f"**{escape(label)} · {format_date(date, frame)}**")
        matching = fits.loc[fits["Label"].eq(label) & fits["Date"].eq(date)]
        if matching.empty:
            st.info("No fit_results entry for this plant and date.")
            continue
        row = matching.iloc[0]
        st.markdown("##### User input")
        st.table(pd.DataFrame([
            {"Parameter": "Measured leaf area", "Value": f"{LEAF_AREA_CM2} (cm²)",
             "Description": "Area of the measured leaf sample"},
            {"Parameter": "Chlorophyll content", "Value": f"{_number(row['Chl'])} (mg/m²)",
             "Description": "Chlorophyll content of the measured leaf"},
        ]))
        st.markdown("##### Parameter estimation")
        if row[success_column] != True:
            st.warning(f"Fit unavailable: {row[message_column]}")
            continue
        st.table(_parameter_table(row, parameters))


def _prepare_model_plot(
    selected: pd.DataFrame, variable: str, fitted_column: str, show_fit: bool,
    date_facets: bool, frame: pd.DataFrame,
) -> pd.DataFrame:
    """Keep original workbook points and separate each date's fitted line."""

    parts = []
    series_definitions = [(variable, "Measured values", "#DC2626", "markers", 1.0)]
    if show_fit:
        # Draw measured points last so the red observations remain visible.
        series_definitions.insert(0, (fitted_column, "YeLP model", "#222222", "lines+markers", 0.45))
    for value_column, legend, color, mode, opacity in series_definitions:
        part = selected.loc[valid_value_mask(selected, value_column) & selected["PAR"].notna()].copy()
        part["Value"] = part[value_column]
        part["SeriesKey"] = part["DateText"] + "|" + legend
        part["LegendLabel"] = legend
        part["ColorKey"] = color
        part["SymbolKey"] = ""
        part["DashKey"] = ""
        part["LineMode"] = mode
        part["MarkerOpacity"] = opacity
        part["FacetRow"] = ""
        part["FacetColumn"] = part["Date"].map(lambda date: format_date(date, frame)) if date_facets else ""
        part["HoverText"] = [
            f"<b>{escape(str(row.Label))}</b><br>{legend}<br>"
            f"Date: {format_date(row.Date, frame)}<br>"
            f"Water: {escape(str(row.Water))} · Nitrogen: {escape(str(row.Nitrogen))}<br>"
            f"PAR: {_number(row.PAR)}<br>Value: {_number(row.Value)}"
            for row in part.itertuples()
        ]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _toggle_fit(key: str) -> None:
    st.session_state[key] = not st.session_state.get(key, False)


def render_model_validation(crop: str, path: Path) -> None:
    """Render crop-specific filters, a fit toggle, parameters, and downloads."""

    st.markdown(f"### {crop}")
    if not path.exists():
        st.error(f"Data file not found: {path.name}")
        return
    try:
        frame, fits = load_model_workbook(str(path.resolve()), *file_signature(path))
    except (OSError, DataValidationError) as exc:
        st.error(str(exc))
        return
    prefix = _state_key(crop, "model_validation")
    variables = [column for column in MODEL_VARIABLES if valid_value_mask(frame, column).any()]
    if not variables:
        st.error("No valid ETR(II) or Y(II) measurements are available.")
        return
    variable = _single_choice(
        "Response variable", variables, _state_key(prefix, "variable"), horizontal=True,
        format_func=lambda column: MODEL_VARIABLES[column][0],
    )
    variable_label, fitted_column, ratio_column, success_column, _ = MODEL_VARIABLES[variable]
    st.caption(
        f"{len(frame):,} records · {frame['Label'].nunique():,} plants · "
        f"{frame['Date'].nunique():,} dates · source: {path.name}"
    )
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("#### Filters")
    valid = frame.loc[valid_value_mask(frame, variable)]
    columns = st.columns(3)
    with columns[0]:
        water = _single_choice("Water", ordered_values(valid["Water"], WATER_ORDER), _state_key(prefix, "water"))
    valid = valid.loc[valid["Water"].eq(water)]
    with columns[1]:
        nitrogen = _single_choice("Nitrogen", ordered_values(valid["Nitrogen"], NITROGEN_ORDER), _state_key(prefix, "nitrogen"))
    valid = valid.loc[valid["Nitrogen"].eq(nitrogen)]
    with columns[2]:
        unit = _single_choice("Block", ordered_values(valid["PlantUnit"]), _state_key(prefix, "plant_unit"))
    valid = valid.loc[valid["PlantUnit"].eq(unit)]
    labels = ordered_values(valid["Label"])
    if not labels:
        st.info("No plant matches the current treatment and block selection.")
        return
    with columns[2]:
        if len(labels) > 1:
            label = _single_choice("Label", labels, _state_key(prefix, "label"))
        else:
            label = labels[0]
            st.caption(f"Label: {label}")
    dates = _date_controls(frame, variable, frame, prefix, include_empty_dates=True)
    if not dates:
        st.info("Select at least one date with valid measurements.")
        return
    presentation = "Single date"
    if len(dates) > 1:
        presentation = _single_choice(
            "Date display", ["Date facets", "Overlay dates"],
            _state_key(prefix, "date_display"), horizontal=True,
        )
    selected = frame.loc[
        frame["Water"].eq(water) & frame["Nitrogen"].eq(nitrogen)
        & frame["PlantUnit"].eq(unit) & frame["Label"].eq(label) & frame["Date"].isin(dates)
    ].sort_values(["Date", "PAR"], kind="stable")
    measured = valid_value_mask(selected, variable) & selected["PAR"].notna()
    if not measured.any():
        st.info("No valid measurements remain for this plant and date selection.")
        return
    selection = (variable, label, tuple(sorted(dates)), presentation)
    fit_key = _state_key(prefix, "show_fit")
    selection_key = _state_key(prefix, "fit_selection")
    if st.session_state.get(selection_key) != selection:
        st.session_state[fit_key] = False
        st.session_state[selection_key] = selection
    st.button(
        "Fit YeLP model", key=_state_key(prefix, "fit_button"),
        on_click=_toggle_fit, args=(fit_key,),
        type="primary" if st.session_state.get(fit_key) else "secondary",
        help="Show or hide the YeLP fitted values and parameters from the workbook.",
    )
    show_fit = st.session_state.get(fit_key, False)
    st.markdown(f"#### {crop} {variable_label}")
    st.caption(f"{label} · {_date_summary(dates, frame)}")
    plot_selection = selected.copy()
    if show_fit:
        successful_dates = fits.loc[fits["Label"].eq(label) & fits[success_column].eq(True), "Date"]
        plot_selection.loc[~plot_selection["Date"].isin(successful_dates), fitted_column] = np.nan
    date_facets = presentation == "Date facets"
    plot_data = _prepare_model_plot(plot_selection, variable, fitted_column, show_fit, date_facets, frame)
    spec = PlotSpec(
        all_data=plot_data, visible_data=plot_data,
        title=f"{crop} {variable_label}<br>{label} · {_date_summary(dates, frame)}",
        y_label=axis_label(variable_label), value_column="Value", error_column=None,
        line_mode="markers", color_map={"#DC2626": "#DC2626", "#222222": "#222222"},
        symbol_order=[""], dash_order=[""], facet_rows=[],
        facet_columns=[format_date(date, frame) for date in sorted(dates)] if date_facets else [],
    )
    invalid_count = int((~measured).sum())
    if invalid_count:
        st.warning(f"Excluded {invalid_count:,} record(s) with missing or invalid measured values or PAR from the measured plot.")
    if show_fit:
        plot_column, parameters_column = st.columns([2, 1], gap="large")
        with plot_column:
            _render_chart(spec, prefix, show_title=False)
        with parameters_column:
            _render_parameters(fits, dates, label, variable, frame)
    else:
        _render_chart(spec, prefix, show_title=False)
    export = selected[[
        "Date", "DAD", "Label", "Water", "Nitrogen", "Block", "PAR",
        variable, fitted_column, ratio_column,
    ]].copy()
    export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")
    export = export.rename(columns={variable: variable_label, ratio_column: "Nk/N0"})
    _render_downloads(spec, export.reset_index(drop=True), _slug(f"{crop}-{variable_label}-{label}-model-validation"), prefix)
