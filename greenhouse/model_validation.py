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
from greenhouse.plotting import PlotAnnotation, PlotSpec, axis_label
from greenhouse.statistics import aggregate_daily_treatment
from greenhouse.query_state import restore_query_variable, sync_query_selection
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
    ("PAR_sat", "PAR<sub>sat</sub>", PAR_UNIT, "Saturation light intensity"),
)
ETR_PARAMETERS = (
    ("alpha_e", "α<sub>e</sub>", "µmol e⁻", "Initial slope"),
    ("beta_e", "β<sub>e</sub>", INVERSE_PAR_UNIT, "Photoinhibition coefficient"),
    ("gamma_e", "γ<sub>e</sub>", INVERSE_PAR_UNIT, "Saturation coefficient"),
    ("gi_gk_ETR", "g<sub>i</sub>/g<sub>k</sub>", "", "Ratio of ground-state to excited-state energy level degeneracies"),
    *COMMON_PARAMETERS,
    ("ETR_max", "ETR<sub>max</sub>", "µmol e⁻ m⁻² s⁻¹", "Maximum electron transport rate"),
    ("sigma_ik", "σ<sub>ik</sub>", "m²", "Intrinsic light absorption cross section"),
    ("tau_min", "τ<sub>min</sub>", "s", "Minimum average lifetime of pigment molecules in the excited state"),
    ("N0", "N<sub>0</sub>", "", "Number of light harvesting pigments"),
    ("ETR_R2", "R<sup>2</sup>", "", "Determination coefficient"),
    ("ETR_rmse", "RMSE", "µmol e⁻ m⁻² s⁻¹", "Root mean squared error"),
)
YII_PARAMETERS = (
    ("beta_e_YII", "β<sub>e</sub>", INVERSE_PAR_UNIT, "Photoinhibition coefficient"),
    ("gamma_e_YII", "γ<sub>e</sub>", INVERSE_PAR_UNIT, "Saturation coefficient"),
    ("gi_gk_YII", "g<sub>i</sub>/g<sub>k</sub>", "", "Ratio of ground-state to excited-state energy level degeneracies"),
    *COMMON_PARAMETERS,
    ("YIImax_Ye", "Y(II)<sub>max</sub>", "", "Maximum effective quantum yield of photosystem II"),
    ("YII_Ye_R2", "R<sup>2</sup>", "", "Determination coefficient"),
    ("YII_Ye_rmse", "RMSE", "", "Root mean squared error"),
)
ANNOTATION_FIELDS = {
    "ETRII": ("alpha_e", "beta_e", "gamma_e", "PAR_sat", "ETR_max", "ETR_R2"),
    "YII": ("beta_e_YII", "gamma_e_YII", "PAR_sat", "YII_Ye_R2"),
}
ANNOTATION_MATH = {
    "alpha_e": r"\alpha_{\mathrm{e}}",
    "beta_e": r"\beta_{\mathrm{e}}",
    "gamma_e": r"\gamma_{\mathrm{e}}",
    "beta_e_YII": r"\beta_{\mathrm{e}}",
    "gamma_e_YII": r"\gamma_{\mathrm{e}}",
    "PAR_sat": r"\mathrm{PAR}_{\mathrm{sat}}",
    "ETR_max": r"\mathrm{ETR}_{\mathrm{max}}",
    "ETR_R2": r"R^{2}",
    "YII_Ye_R2": r"R^{2}",
}


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
    missing_curves = sorted({"DAD", "Block", "Chl", *MODEL_NUMERIC_COLUMNS}.difference(curves.columns))
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


def _model_number(value: object, precision: int = 3) -> str:
    """Limit significant digits while retaining very small model coefficients."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Not available"
    return f"{numeric:.{precision}g}" if np.isfinite(numeric) else "Not estimable"


def _parameter_table(row: pd.Series, parameters: tuple) -> pd.DataFrame:
    records = []
    for column, symbol, unit, description in parameters:
        records.append({
            "Parameter": symbol, "Value": _model_number(row.get(column)),
            "Unit": unit, "Description": description,
        })
    return pd.DataFrame(records)


def _render_parameter_table(table: pd.DataFrame) -> None:
    """Render normal English-letter subscripts and a separate unit column."""

    headings = "".join(f"<th scope='col'>{escape(column)}</th>" for column in table.columns)
    rows = []
    for record in table.to_dict("records"):
        # Parameter symbols are trusted constants; workbook-derived text is escaped.
        cells = "".join(
            f"<td>{value if column == 'Parameter' else escape(str(value))}</td>"
            for column, value in record.items()
        )
        rows.append(f"<tr>{cells}</tr>")
    st.html(
        "<style>"
        ".model-parameter-table { overflow-x: auto; }"
        ".model-parameter-table table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }"
        ".model-parameter-table th, .model-parameter-table td "
        "{ text-align: left; vertical-align: top; padding: 0.5rem; border-bottom: 1px solid #dce5df; }"
        ".model-parameter-table th { font-weight: 600; }"
        ".model-parameter-table td:nth-child(2) { white-space: nowrap; font-variant-numeric: tabular-nums; }"
        ".model-parameter-table sub, .model-parameter-table sup { font-size: 0.75em; }"
        "</style>"
        f"<div class='model-parameter-table'><table><thead><tr>{headings}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_parameters(
    fits: pd.DataFrame, dates: list[pd.Timestamp], label: str, variable: str, frame: pd.DataFrame,
    *, mean_fits: pd.DataFrame | None = None, mean_by_date: bool = False,
    treatment: bool = False,
) -> None:
    st.markdown("#### Parameters")
    if mean_fits is not None:
        if mean_fits.empty:
            st.info("No fitted values are available for the selected dates.")
            return
        st.caption(
            "Arithmetic means across available fits, with equal weight per date; "
            "missing parameter values are omitted."
        )
        excluded = len(dates) - mean_fits["Date"].nunique()
        if excluded:
            st.caption(f"{excluded} selected date(s) without an available fit are excluded.")
        if mean_by_date:
            for date in sorted(dates):
                daily = mean_fits.loc[mean_fits["Date"].eq(date)]
                st.markdown(f"**{escape(label)} · {format_date(date, frame)}**")
                if daily.empty:
                    st.info("No fitted values are available for this date.")
                    continue
                st.caption(_mean_fit_heading(daily, treatment=treatment))
                _render_parameter_values(_mean_parameter_values(daily, variable), variable)
        else:
            heading = _mean_fit_heading(mean_fits, treatment=treatment)
            st.markdown(f"**{escape(label)} · {heading}**")
            _render_parameter_values(_mean_parameter_values(mean_fits, variable), variable)
        return
    for date in sorted(dates):
        st.markdown(f"**{escape(label)} · {format_date(date, frame)}**")
        matching = fits.loc[fits["Label"].eq(label) & fits["Date"].eq(date)]
        if matching.empty:
            st.info("No fit_results entry for this plant and date.")
            continue
        _render_parameter_values(matching.iloc[0], variable)


def _render_parameter_values(row: pd.Series, variable: str) -> None:
    parameters = ETR_PARAMETERS if variable == "ETRII" else YII_PARAMETERS
    _, _, _, success_column, message_column = MODEL_VARIABLES[variable]
    st.markdown("##### User input")
    _render_parameter_table(pd.DataFrame([
        {"Parameter": "Leaf area", "Value": _model_number(LEAF_AREA_CM2), "Unit": "cm²",
         "Description": "Circular region illuminated by the Dual-PAM optical fiber."},
        {"Parameter": "Chlorophyll content", "Value": _model_number(row["Chl"]), "Unit": "mg/m²",
         "Description": "Mean values of 3 spots of the measured leaf, measured by LSA"},
    ]))
    st.markdown("##### Parameter estimation")
    if row.get(success_column, True) != True:
        st.warning(f"Fit unavailable: {row[message_column]}")
        return
    _render_parameter_table(_parameter_table(row, parameters))


def _matching_model_fits(
    fits: pd.DataFrame, selected: pd.DataFrame, variable: str,
) -> pd.DataFrame:
    """Return one successful fit per visible plant/date curve."""

    _, fitted_column, _, success_column, _ = MODEL_VARIABLES[variable]
    visible = selected.loc[valid_value_mask(selected, fitted_column) & selected["PAR"].notna()]
    curves = visible[["Label", "Date"]].drop_duplicates()
    matching = fits.merge(curves, on=["Label", "Date"]).sort_values("Date")
    return matching.loc[matching[success_column].eq(True)]


def _mean_parameter_values(fits: pd.DataFrame, variable: str) -> pd.Series:
    """Average plants within dates, then give each fitted date equal weight."""

    parameters = ETR_PARAMETERS if variable == "ETRII" else YII_PARAMETERS
    columns = ["Chl", *(column for column, _, _, _ in parameters)]
    numeric = fits[columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.groupby(fits["Date"]).mean().mean()


def _mean_fit_heading(fits: pd.DataFrame, *, treatment: bool = False) -> str:
    dates = fits["Date"].nunique()
    date_text = f"{dates} {'date' if dates == 1 else 'dates'}"
    if treatment:
        plants = fits["Label"].nunique()
        plant_text = f"{plants} {'plant' if plants == 1 else 'plants'}"
        return f"Mean ({plant_text} · {date_text})"
    return f"Mean ({date_text})"


def _model_annotations(
    fits: pd.DataFrame, selected: pd.DataFrame, variable: str,
    date_facets: bool, frame: pd.DataFrame, *, mean_dates: bool = False,
    mean_plants: bool = False,
) -> list[PlotAnnotation]:
    """Summarize successful fits for the curves actually shown in each panel."""

    matching = _matching_model_fits(fits, selected, variable)
    if matching.empty:
        return []
    if mean_dates:
        summaries = [("", _mean_parameter_values(matching, variable),
                      _mean_fit_heading(matching, treatment=mean_plants))]
    elif mean_plants:
        summaries = [
            (format_date(date, frame) if date_facets else "",
             _mean_parameter_values(daily, variable),
             _mean_fit_heading(daily, treatment=True))
            for date, daily in matching.groupby("Date", sort=True)
        ]
    else:
        summaries = [
            (format_date(row["Date"], frame) if date_facets else "", row, "")
            for _, row in matching.iterrows()
        ]
    parameters = ETR_PARAMETERS if variable == "ETRII" else YII_PARAMETERS
    symbols = {column: symbol for column, symbol, _, _ in parameters}
    sections: dict[str, list[tuple[str, str]]] = {}
    for panel, row, heading in summaries:
        html_lines, static_lines = [], []
        for column in ANNOTATION_FIELDS[variable]:
            value = _model_number(row[column], precision=3)
            html_lines.append(f"{symbols[column]} = {value}")
            static_lines.append(f"${ANNOTATION_MATH[column]}$ = {value}")
        if heading:
            html_lines.insert(0, f"<b>{heading}</b>")
            static_lines.insert(0, heading)
        sections.setdefault(panel, []).append(("<br>".join(html_lines), "\n".join(static_lines)))
    return [
        PlotAnnotation(
            html_text="<br><br>".join(html for html, _ in summaries),
            text="\n\n".join(text for _, text in summaries), facet_column=panel,
        )
        for panel, summaries in sections.items()
    ]


def _mean_treatment_data(
    selected: pd.DataFrame, variable: str, fitted_column: str, ratio_column: str,
) -> pd.DataFrame:
    """Prepare daily treatment means using Explore's plant aggregation rules."""

    groups = ["Date", "Water", "Nitrogen", "PAR"]
    summaries = []
    for column in (variable, fitted_column, ratio_column):
        daily, _, _ = aggregate_daily_treatment(selected, column)
        if daily.empty:
            summary = selected[groups].iloc[:0].copy()
            for name in (column, f"{column}_SD", f"{column}_n"):
                summary[name] = pd.Series(dtype=float)
        else:
            summary = daily[groups + ["Mean", "SD", "n"]].rename(columns={
                "Mean": column, "SD": f"{column}_SD", "n": f"{column}_n",
            })
        summaries.append(summary)
    result = summaries[0]
    for summary in summaries[1:]:
        result = result.merge(summary, on=groups, how="outer", validate="one_to_one")
    result["DateText"] = pd.to_datetime(result["Date"]).dt.strftime("%Y-%m-%d")
    result["DAD"] = result["Date"].map(selected.groupby("Date")["DAD"].first())
    result["Label"] = result["Water"] + " × " + result["Nitrogen"] + " mean"
    return result.sort_values(["Date", "PAR"], kind="stable").reset_index(drop=True)


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
            f"PAR: {_number(row.PAR)}<br>Value: {_number(row.Value)}"
            for row in part.itertuples()
        ]
        if f"{value_column}_n" in part:
            part["HoverText"] += "<br>Plants: " + part[f"{value_column}_n"].astype("Int64").astype(str)
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
    variable_key = _state_key(prefix, "variable")
    restore_query_variable(crop, variable_key, {column: MODEL_VARIABLES[column][0] for column in variables})
    variable = _single_choice(
        "Response variable", variables, variable_key, horizontal=True,
        format_func=lambda column: MODEL_VARIABLES[column][0],
    )
    variable_label, fitted_column, ratio_column, success_column, _ = MODEL_VARIABLES[variable]
    sync_query_selection(crop, "Model Validation", variable_label)
    st.caption(
        f"{len(frame):,} records · {frame['Label'].nunique():,} plants · "
        f"{frame['Date'].nunique():,} dates · source: {path.name}"
    )
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    plot_kind = _single_choice(
        "Validation view", ["Single plant", "Treatment curve"],
        _state_key(prefix, "view"), horizontal=True,
    )
    treatment = plot_kind == "Treatment curve"
    if treatment:
        prefix = _state_key(crop, "model_validation", "treatment_curve")
    st.markdown("#### Filters")
    valid = frame.loc[valid_value_mask(frame, variable)]
    columns = st.columns(2 if treatment else 3)
    with columns[0]:
        water = _single_choice("Water", ordered_values(valid["Water"], WATER_ORDER), _state_key(prefix, "water"))
    valid = valid.loc[valid["Water"].eq(water)]
    with columns[1]:
        nitrogen = _single_choice("Nitrogen", ordered_values(valid["Nitrogen"], NITROGEN_ORDER), _state_key(prefix, "nitrogen"))
    valid = valid.loc[valid["Nitrogen"].eq(nitrogen)]
    scoped = frame.loc[frame["Water"].eq(water) & frame["Nitrogen"].eq(nitrogen)]
    if treatment:
        label = f"{water} × {nitrogen}"
        dates = _date_controls(scoped, variable, frame, prefix)
    else:
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
        scoped = scoped.loc[scoped["PlantUnit"].eq(unit) & scoped["Label"].eq(label)]
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
    selected = scoped.loc[scoped["Date"].isin(dates)].sort_values(["Date", "PAR"], kind="stable")
    measured = valid_value_mask(selected, variable) & selected["PAR"].notna()
    if not measured.any():
        st.info("No valid measurements remain for this selection.")
        return
    selection = (variable, plot_kind, label, tuple(sorted(dates)), presentation)
    fit_key = _state_key(prefix, "show_fit")
    selection_key = _state_key(prefix, "fit_selection")
    if st.session_state.get(selection_key) != selection:
        st.session_state[fit_key] = False
        st.session_state[selection_key] = selection
    title_key = _state_key(crop, "model_validation", "plot_title")
    st.html(
        f"<style>.st-key-{title_key} h4 "
        "{ margin: 0 !important; padding: 0 !important; }"
        f".st-key-{title_key} [data-testid='stButton'] "
        "{ position: relative; top: 12px; }</style>"
    )
    with st.container(
        key=title_key, horizontal=True, vertical_alignment="center", gap="medium"
    ):
        st.markdown(f"#### {crop} {variable_label}", width="content")
        st.button(
            "**Fit YeLP model**", key=_state_key(prefix, "fit_button"),
            on_click=_toggle_fit, args=(fit_key,),
            type="primary" if st.session_state.get(fit_key) else "secondary",
            help="Show or hide the YeLP fitted values and parameters from the workbook.",
        )
    show_fit = st.session_state.get(fit_key, False)
    st.caption(f"{label} · {_date_summary(dates, frame)}")
    plot_selection = selected.copy()
    if show_fit:
        successful = fits.loc[fits[success_column].eq(True), ["Label", "Date"]].assign(FitAvailable=True)
        plot_selection = plot_selection.merge(successful, on=["Label", "Date"], how="left", validate="many_to_one")
        plot_selection.loc[~plot_selection["FitAvailable"].eq(True), [fitted_column, ratio_column]] = np.nan
    date_facets = presentation == "Date facets"
    display_data = _mean_treatment_data(plot_selection, variable, fitted_column, ratio_column) if treatment else plot_selection
    plot_data = _prepare_model_plot(display_data, variable, fitted_column, show_fit, date_facets, frame)
    spec = PlotSpec(
        all_data=plot_data, visible_data=plot_data,
        title=f"{crop} {variable_label}<br>{label} · {_date_summary(dates, frame)}",
        y_label=axis_label(variable_label), value_column="Value", error_column=None,
        line_mode="markers", color_map={"#DC2626": "#DC2626", "#222222": "#222222"},
        symbol_order=[""], dash_order=[""], facet_rows=[],
        facet_columns=[format_date(date, frame) for date in sorted(dates)] if date_facets else [],
        annotations=_model_annotations(
            fits, plot_selection, variable, date_facets, frame,
            mean_dates=presentation == "Overlay dates",
            mean_plants=treatment,
        ) if show_fit else [],
        include_y_zero=True,
    )
    invalid_count = int((~measured).sum())
    if invalid_count:
        st.warning(f"Excluded {invalid_count:,} record(s) with missing or invalid measured values or PAR from the measured plot.")
    if show_fit:
        plot_column, parameters_column = st.columns([2, 1], gap="large")
        with plot_column:
            _render_chart(spec, prefix, show_title=False)
        with parameters_column:
            _render_parameters(
                fits, dates, label, variable, frame,
                mean_fits=_matching_model_fits(fits, plot_selection, variable)
                if treatment or presentation == "Overlay dates" else None,
                mean_by_date=treatment and date_facets, treatment=treatment,
            )
    else:
        _render_chart(spec, prefix, show_title=False)
    if treatment:
        export = display_data[[
            "Date", "DAD", "Water", "Nitrogen", "PAR", variable, fitted_column,
            ratio_column, f"{variable}_SD", f"{variable}_n", f"{fitted_column}_n",
        ]].copy().rename(columns={
            f"{variable}_SD": f"{variable_label} SD",
            f"{variable}_n": "Measured plants", f"{fitted_column}_n": "Fitted plants",
        })
    else:
        export = selected[[
            "Date", "DAD", "Label", "Water", "Nitrogen", "Block", "Chl", "PAR",
            variable, fitted_column, ratio_column,
        ]].copy()
    export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")
    export = export.rename(columns={variable: variable_label, ratio_column: "Nk/N0"})
    _render_downloads(spec, export.reset_index(drop=True), _slug(f"{crop}-{variable_label}-{label}-model-validation"), prefix)
