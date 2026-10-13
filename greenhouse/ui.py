"""Streamlit controls and plot preparation shared by both crop panels."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

from greenhouse.data import (
    DataValidationError,
    VARIABLE_LABELS,
    available_variables,
    file_signature,
    format_date,
    load_workbook,
    ordered_values,
    valid_value_mask,
)
from greenhouse.plotting import (
    NITROGEN_COLORS,
    WATER_COLORS,
    PlotSpec,
    axis_label,
    create_plotly_figure,
    create_static_export,
    stable_color_map,
)
from greenhouse.statistics import (
    aggregate_daily_treatment,
    aggregate_individual,
    aggregate_period_treatment,
)


WATER_ORDER = ("WW", "DT", "RW")
NITROGEN_ORDER = ("N0", "N1", "N2")
REWATERING_DATE = pd.Timestamp("2024-12-06")
DATE_COLORS = (
    "#236A8C",
    "#C65D21",
    "#3B7D4F",
    "#8B4F9F",
    "#B47D00",
    "#A63D5B",
    "#247B78",
    "#8C4A32",
    "#4D5FA3",
    "#6F7035",
)
LINE_MODE_LABELS = {
    "Points + lines": "lines+markers",
    "Points": "markers",
    "Lines": "lines",
}


def _state_key(*parts: object) -> str:
    return "greenhouse__" + "__".join(str(part).lower().replace(" ", "_") for part in parts)


def _single_choice(
    label: str,
    options: list,
    key: str,
    *,
    default=None,
    format_func=None,
    horizontal: bool = False,
    help_text: str | None = None,
):
    if not options:
        return None
    selected_default = default if default in options else options[0]
    if st.session_state.get(key) not in options:
        st.session_state[key] = selected_default
    if horizontal:
        return st.radio(
            label,
            options,
            key=key,
            format_func=format_func or str,
            horizontal=True,
            help=help_text,
        )
    return st.selectbox(
        label,
        options,
        key=key,
        format_func=format_func or str,
        help=help_text,
    )


def _sync_multiselect_state(
    key: str,
    options: list,
    *,
    default_all: bool,
    maximum: int | None = None,
) -> None:
    previous_options_key = f"{key}__available"
    previous_options = st.session_state.get(previous_options_key, [])
    current = st.session_state.get(key)
    if current is None:
        selected = list(options if default_all else [])
    else:
        was_all = bool(previous_options) and set(current) == set(previous_options)
        if was_all and default_all:
            selected = list(options)
        else:
            selected = [value for value in current if value in options]
    if maximum is not None:
        selected = selected[:maximum]
    st.session_state[key] = selected
    st.session_state[previous_options_key] = list(options)


def _multiple_choice(
    label: str,
    options: list,
    key: str,
    *,
    default_all: bool = True,
    maximum: int | None = None,
    format_func=None,
    help_text: str | None = None,
) -> list:
    _sync_multiselect_state(key, options, default_all=default_all, maximum=maximum)
    return st.multiselect(
        label,
        options,
        key=key,
        max_selections=maximum,
        format_func=format_func or str,
        help=help_text,
    )


def _sync_date_state(key: str, options: list[pd.Timestamp]) -> None:
    previous_options_key = f"{key}__available"
    previous_options = st.session_state.get(previous_options_key, [])
    current = st.session_state.get(key)
    if current is None:
        selected = options[-1:]
    elif not current:
        selected = []
    else:
        was_all = bool(previous_options) and set(current) == set(previous_options)
        selected = list(options) if was_all else [value for value in current if value in options]
        if not selected:
            selected = options[-1:]
    st.session_state[key] = selected
    st.session_state[previous_options_key] = list(options)


def _set_selected_dates(date_key: str, selected_dates: list[pd.Timestamp]) -> None:
    st.session_state[date_key] = list(selected_dates)


def _date_period_styles(dates: list[pd.Timestamp], crop_frame: pd.DataFrame) -> None:
    before_labels = [format_date(date, crop_frame) for date in dates if date < REWATERING_DATE]
    after_labels = [format_date(date, crop_frame) for date in dates if date >= REWATERING_DATE]

    def selectors(labels: list[str]) -> str:
        return ",".join(
            f'[data-baseweb="tag"]:has(span[title="{label}"])' for label in labels
        )

    before_selectors = selectors(before_labels)
    after_selectors = selectors(after_labels)
    rules: list[str] = []
    if before_selectors:
        rules.append(
            f"{before_selectors} {{ background-color: #D9EAF4 !important; color: #214C68 !important; }}"
        )
        rules.append(f"{before_selectors} span {{ color: #214C68 !important; }}")
    if after_selectors:
        rules.append(
            f"{after_selectors} {{ background-color: #F6E4C9 !important; color: #704515 !important; }}"
        )
        rules.append(f"{after_selectors} span {{ color: #704515 !important; }}")
    if rules:
        st.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)


def _date_controls(
    frame: pd.DataFrame,
    variable: str,
    crop_frame: pd.DataFrame,
    prefix: str,
) -> list[pd.Timestamp]:
    valid = frame.loc[valid_value_mask(frame, variable)]
    dates = sorted(pd.Timestamp(value).normalize() for value in valid["Date"].unique())
    if not dates:
        return []

    format_func = lambda value: format_date(pd.Timestamp(value), crop_frame)
    date_key = _state_key(prefix, "dates")
    _sync_date_state(date_key, dates)
    before_dates = [date for date in dates if date < REWATERING_DATE]
    after_dates = [date for date in dates if date >= REWATERING_DATE]
    if before_dates and after_dates:
        _date_period_styles(dates, crop_frame)
    selected = st.multiselect(
        "Dates",
        dates,
        key=date_key,
        format_func=format_func,
    )
    if before_dates and after_dates:
        period_columns = st.columns(2)
        with period_columns[0]:
            st.button(
                "Before rewatering",
                key=_state_key(prefix, "before_rewatering_dates_button"),
                on_click=_set_selected_dates,
                args=(date_key, before_dates),
                width="stretch",
            )
        with period_columns[1]:
            st.button(
                "After rewatering",
                key=_state_key(prefix, "after_rewatering_dates_button"),
                on_click=_set_selected_dates,
                args=(date_key, after_dates),
                width="stretch",
            )
    if any(pd.Timestamp(date) < REWATERING_DATE for date in selected):
        st.caption("Before rewatering, interpret the Water label RW as DT.")
    return [pd.Timestamp(value) for value in selected]


def _date_facets(
    selected_dates: list[pd.Timestamp],
    crop_frame: pd.DataFrame,
) -> tuple[list[pd.Timestamp], list[str]]:
    ordered_dates = sorted(selected_dates)
    return ordered_dates, [format_date(date, crop_frame) for date in ordered_dates]


def _date_summary(dates: list[pd.Timestamp], crop_frame: pd.DataFrame) -> str:
    ordered_dates = sorted(dates)
    if not ordered_dates:
        return "No dates"
    if len(ordered_dates) == 1:
        return format_date(ordered_dates[0], crop_frame)
    return (
        f"{format_date(ordered_dates[0], crop_frame)} to "
        f"{format_date(ordered_dates[-1], crop_frame)} ({len(ordered_dates)} dates)"
    )


def _period_label(dates: list[pd.Timestamp]) -> str:
    ordered_dates = sorted(dates)
    if len(ordered_dates) == 1:
        return ordered_dates[0].strftime("%Y-%m-%d")
    return f"{ordered_dates[0].strftime('%Y-%m-%d')} to {ordered_dates[-1].strftime('%Y-%m-%d')}"


def _treatment_text(frame: pd.DataFrame) -> pd.Series:
    return frame["Water"].astype(str) + " × " + frame["Nitrogen"].astype(str)


def _number(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not np.isfinite(numeric):
        return "Not estimable"
    return f"{numeric:.6g}"


def _prepare_individual_plot(
    plants: pd.DataFrame,
    *,
    crop_frame: pd.DataFrame,
    color_by: str,
    symbol_by: str,
    date_facets: bool,
    overlay_dates: bool,
    single_plant: bool,
) -> pd.DataFrame:
    plot = plants.copy()
    date_labels = {date: format_date(pd.Timestamp(date), crop_frame) for date in plot["Date"].unique()}
    plot["DateDisplay"] = plot["Date"].map(date_labels)
    plot["Treatment"] = _treatment_text(plot)
    plot["FacetRow"] = ""
    plot["FacetColumn"] = plot["DateDisplay"] if date_facets else ""

    if single_plant and overlay_dates:
        plot["SeriesKey"] = plot["DateText"]
        plot["LegendLabel"] = plot["DateDisplay"]
        plot["ColorKey"] = plot["DateText"]
        plot["SymbolKey"] = ""
        plot["DashKey"] = plot["DateText"]
    elif single_plant and date_facets:
        plot["SeriesKey"] = plot["DateText"]
        plot["LegendLabel"] = plot["DateDisplay"]
        plot["ColorKey"] = plot["DateText"]
        plot["SymbolKey"] = ""
        plot["DashKey"] = ""
    else:
        plot["SeriesKey"] = plot["Label"]
        plot["LegendLabel"] = plot["Label"]
        plot["ColorKey"] = plot[color_by]
        plot["SymbolKey"] = "" if symbol_by == "None" else plot[symbol_by]
        plot["DashKey"] = ""

    plot["HoverText"] = [
        "<br>".join(
            [
                f"<b>{escape(str(row.Label))}</b>",
                f"Date: {escape(str(row.DateDisplay))}",
                f"Treatment: {escape(str(row.Treatment))}",
                f"PAR: {_number(row.PAR)}",
                f"Value: {_number(row.Value)}",
                f"Merged records: {int(row.RawRecords):,}",
            ]
        )
        for row in plot.itertuples()
    ]
    return plot


def _prepare_summary_plot(
    summary: pd.DataFrame,
    *,
    crop_frame: pd.DataFrame,
    dates: list[pd.Timestamp],
    presentation: str,
    comparison_factor: str | None,
    facet_factor: str | None,
) -> pd.DataFrame:
    plot = summary.copy()
    is_period = presentation == "Period summary"
    period = _period_label(dates)
    if is_period:
        plot["DateDisplay"] = period
        plot["DateText"] = period
    else:
        date_labels = {date: format_date(pd.Timestamp(date), crop_frame) for date in plot["Date"].unique()}
        plot["DateDisplay"] = plot["Date"].map(date_labels)

    plot["Treatment"] = _treatment_text(plot)
    plot["FacetRow"] = (
        facet_factor + ": " + plot[facet_factor].astype(str) if facet_factor else ""
    )
    plot["FacetColumn"] = plot["DateDisplay"] if presentation == "Date facets" else ""

    if comparison_factor:
        base = plot[comparison_factor].astype(str)
        if presentation == "Overlay dates":
            plot["SeriesKey"] = base + "|" + plot["DateText"].astype(str)
            plot["LegendLabel"] = base + " · " + plot["DateDisplay"].astype(str)
            plot["DashKey"] = plot["DateText"].astype(str)
        else:
            plot["SeriesKey"] = base
            plot["LegendLabel"] = base
            plot["DashKey"] = ""
        plot["ColorKey"] = base
    elif presentation == "Overlay dates":
        plot["SeriesKey"] = plot["DateText"].astype(str)
        plot["LegendLabel"] = plot["DateDisplay"].astype(str)
        plot["ColorKey"] = plot["DateText"].astype(str)
        plot["DashKey"] = plot["DateText"].astype(str)
    else:
        plot["SeriesKey"] = plot["Treatment"]
        plot["LegendLabel"] = plot["Treatment"]
        plot["ColorKey"] = plot["Water"].astype(str)
        plot["DashKey"] = ""
    plot["SymbolKey"] = ""

    context_label = "Period" if is_period else "Date"
    plot["HoverText"] = [
        "<br>".join(
            [
                f"<b>{escape(str(row.Treatment))}</b>",
                f"{context_label}: {escape(str(row.DateDisplay))}",
                f"PAR: {_number(row.PAR)}",
                f"Mean: {_number(row.Mean)}",
                f"SD: {_number(row.SD)}",
                f"n: {int(row.n):,}",
            ]
        )
        for row in plot.itertuples()
    ]
    return plot


def _style_maps(frame: pd.DataFrame) -> tuple[dict[str, str], list[str], list[str]]:
    labels = ordered_values(frame["Label"])
    ordered_dates = [pd.Timestamp(value) for value in sorted(frame["Date"].unique())]
    dates = [value.strftime("%Y-%m-%d") for value in ordered_dates]
    waters = ordered_values(frame["Water"], WATER_ORDER)
    nitrogens = ordered_values(frame["Nitrogen"], NITROGEN_ORDER)
    colors = stable_color_map(waters, WATER_COLORS)
    colors.update(stable_color_map(nitrogens, NITROGEN_COLORS))
    colors.update(
        {
            value.strftime("%Y-%m-%d"): DATE_COLORS[index % len(DATE_COLORS)]
            for index, value in enumerate(ordered_dates)
        }
    )
    colors.update(stable_color_map(labels))
    identity = ordered_values(frame["PlantUnit"])
    symbols = identity + waters + nitrogens + [""]
    return colors, symbols, dates


def _export_table(plot_data: pd.DataFrame, variable_label: str, identity_column: str) -> pd.DataFrame:
    table = plot_data.copy()
    if "Date" in table:
        table["Date"] = pd.to_datetime(table["Date"]).dt.strftime("%Y-%m-%d")
    if "Value" in table:
        columns = ["Date"]
        if "DAD" in table:
            columns.append("DAD")
        columns += ["Label", "Water", "Nitrogen", "PlantUnit", "PAR", "Value", "RawRecords"]
        table = table[[column for column in columns if column in table]].rename(
            columns={"PlantUnit": identity_column, "Value": variable_label}
        )
    else:
        columns = []
        if "Date" in table:
            columns.append("Date")
        elif "DateDisplay" in table:
            table["Period"] = table["DateDisplay"]
            columns.append("Period")
        if "DAD" in table:
            columns.append("DAD")
        columns += ["Water", "Nitrogen", "PAR", "Mean", "SD", "n"]
        table = table[[column for column in columns if column in table]]
        table = table.rename(columns={"Mean": f"{variable_label} mean", "SD": f"{variable_label} SD"})
    return table.reset_index(drop=True)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _render_result(
    spec: PlotSpec,
    export_table: pd.DataFrame,
    *,
    invalid_count: int,
    total_count: int,
    filename_stem: str,
    key_prefix: str,
) -> None:
    if invalid_count:
        st.warning(
            f"Excluded {invalid_count:,} of {total_count:,} filtered record(s) because the selected "
            "variable was missing, non-numeric, or infinite."
        )
    else:
        st.caption(f"All {total_count:,} filtered records contain a valid value for this variable.")

    if spec.error_column and "n" in spec.all_data:
        single_plant_points = int(spec.all_data["n"].eq(1).sum())
        if single_plant_points:
            st.info(
                f"SD is not estimable for {single_plant_points:,} treatment point(s) with n = 1; "
                "their means are shown without a ribbon."
            )

    figure = create_plotly_figure(spec)
    plot_config = {
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "responsive": False,
    }
    if len(spec.facet_columns) > 3:
        figure_width = max(1380, 460 * len(spec.facet_columns))
        figure.update_layout(width=figure_width, autosize=False)
        with st.container(
            key=_state_key(key_prefix, "plot_scroller"),
            horizontal=True,
            wrap=False,
            gap=None,
        ):
            st.plotly_chart(
                figure,
                width=figure_width,
                key=_state_key(key_prefix, "interactive_plot"),
                config=plot_config,
            )
    else:
        st.plotly_chart(
            figure,
            width="stretch",
            key=_state_key(key_prefix, "interactive_plot"),
            config=plot_config,
        )

    with st.expander("Plot data and downloads", expanded=False):
        st.caption(
            "The table and CSV include every selected date. PNG and PDF contain the currently visible date page."
        )
        st.dataframe(export_table, width="stretch", hide_index=True)
        csv_data = export_table.to_csv(index=False, float_format="%.10g").encode("utf-8")
        try:
            png_data = create_static_export(spec, "png")
            pdf_data = create_static_export(spec, "pdf")
        except Exception as exc:
            png_data = pdf_data = None
            st.error(f"Static export could not be prepared: {exc}")

        csv_column, png_column, pdf_column = st.columns(3)
        with csv_column:
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=f"{filename_stem}.csv",
                mime="text/csv",
                key=_state_key(key_prefix, "csv_download"),
                width="stretch",
            )
        with png_column:
            if png_data is not None:
                st.download_button(
                    "Download PNG",
                    data=png_data,
                    file_name=f"{filename_stem}.png",
                    mime="image/png",
                    key=_state_key(key_prefix, "png_download"),
                    width="stretch",
                )
        with pdf_column:
            if pdf_data is not None:
                st.download_button(
                    "Download PDF",
                    data=pdf_data,
                    file_name=f"{filename_stem}.pdf",
                    mime="application/pdf",
                    key=_state_key(key_prefix, "pdf_download"),
                    width="stretch",
                )


def _build_spec(
    plot_data: pd.DataFrame,
    *,
    visible_dates: list[pd.Timestamp] | None,
    facet_columns: list[str],
    facet_rows: list[str],
    title: str,
    variable_label: str,
    line_mode: str,
    color_map: dict[str, str],
    symbol_order: list[str],
    dash_order: list[str],
    summary: bool,
    show_ribbon: bool | None = None,
) -> PlotSpec:
    visible = plot_data
    if visible_dates is not None and "Date" in plot_data:
        visible = plot_data.loc[plot_data["Date"].isin(visible_dates)].copy()
    return PlotSpec(
        all_data=plot_data,
        visible_data=visible,
        title=title,
        y_label=axis_label(variable_label),
        value_column="Mean" if summary else "Value",
        error_column="SD" if summary else None,
        line_mode=line_mode,
        color_map=color_map,
        symbol_order=symbol_order,
        dash_order=dash_order,
        facet_rows=facet_rows,
        facet_columns=facet_columns,
        show_ribbon=summary if show_ribbon is None else show_ribbon,
    )


def _render_single_plant(
    crop: str,
    frame: pd.DataFrame,
    variable: str,
    variable_label: str,
    identity_column: str,
    color_map: dict[str, str],
    symbol_order: list[str],
    dash_order: list[str],
) -> None:
    prefix = _state_key(crop, "explore", "single_plant", variable)
    valid = frame.loc[valid_value_mask(frame, variable)]
    filter_columns = st.columns(3)
    water_options = ordered_values(valid["Water"], WATER_ORDER)
    with filter_columns[0]:
        water = _single_choice("Water", water_options, _state_key(prefix, "water"))
    valid = valid.loc[valid["Water"].eq(water)]
    nitrogen_options = ordered_values(valid["Nitrogen"], NITROGEN_ORDER)
    with filter_columns[1]:
        nitrogen = _single_choice("Nitrogen", nitrogen_options, _state_key(prefix, "nitrogen"))
    valid = valid.loc[valid["Nitrogen"].eq(nitrogen)]
    unit_options = ordered_values(valid["PlantUnit"])
    with filter_columns[2]:
        unit = _single_choice(identity_column, unit_options, _state_key(prefix, "plant_unit"))
    valid = valid.loc[valid["PlantUnit"].eq(unit)]
    label_options = ordered_values(valid["Label"])
    if len(label_options) > 1:
        with filter_columns[2]:
            label = _single_choice("Label", label_options, _state_key(prefix, "label"))
    elif label_options:
        label = label_options[0]
        with filter_columns[2]:
            st.caption(f"Label: {label}")
    else:
        st.info("No plant matches the current treatment and block/replicate selection.")
        return

    scoped = frame.loc[
        frame["Water"].eq(water)
        & frame["Nitrogen"].eq(nitrogen)
        & frame["PlantUnit"].eq(unit)
        & frame["Label"].eq(label)
    ]
    with st.container():
        dates = _date_controls(scoped, variable, frame, prefix)
    if not dates:
        st.info("Select at least one date with valid measurements.")
        return

    presentation = "Single date"
    if len(dates) > 1:
        presentation = _single_choice(
            "Date display",
            ["Date facets", "Overlay dates"],
            _state_key(prefix, "date_display"),
            default="Date facets",
            horizontal=True,
        )
    selected = scoped.loc[scoped["Date"].isin(dates)]
    plants, invalid_count, total_count = aggregate_individual(selected, variable)
    if plants.empty:
        st.info("No valid measurements remain for this plant and date selection.")
        return

    date_facets = presentation == "Date facets"
    overlay_dates = presentation == "Overlay dates"
    plot_data = _prepare_individual_plot(
        plants,
        crop_frame=frame,
        color_by="Label",
        symbol_by="None",
        date_facets=date_facets,
        overlay_dates=overlay_dates,
        single_plant=True,
    )
    visible_dates: list[pd.Timestamp] | None = None
    facet_columns: list[str] = []
    if date_facets:
        visible_dates, facet_columns = _date_facets(dates, frame)
    mode_text = "Plant means after duplicate LC records are merged"
    title = (
        f"{crop} · {variable_label}<br>"
        f"{label} · {water} × {nitrogen} · {_date_summary(dates, frame)} · {mode_text}"
    )
    spec = _build_spec(
        plot_data,
        visible_dates=visible_dates,
        facet_columns=facet_columns,
        facet_rows=[],
        title=title,
        variable_label=variable_label,
        line_mode="lines+markers",
        color_map=color_map,
        symbol_order=symbol_order,
        dash_order=dash_order,
        summary=False,
    )
    export = _export_table(plot_data, variable_label, identity_column)
    _render_result(
        spec,
        export,
        invalid_count=invalid_count,
        total_count=total_count,
        filename_stem=_slug(f"{crop}-{variable_label}-{label}-single-plant"),
        key_prefix=prefix,
    )


def _render_treatment_curve(
    crop: str,
    frame: pd.DataFrame,
    variable: str,
    variable_label: str,
    identity_column: str,
    color_map: dict[str, str],
    symbol_order: list[str],
    dash_order: list[str],
) -> None:
    prefix = _state_key(crop, "explore", "treatment_curve", variable)
    valid = frame.loc[valid_value_mask(frame, variable)]
    filter_columns = st.columns(2)
    waters = ordered_values(valid["Water"], WATER_ORDER)
    with filter_columns[0]:
        water = _single_choice("Water", waters, _state_key(prefix, "water"))
    valid = valid.loc[valid["Water"].eq(water)]
    nitrogens = ordered_values(valid["Nitrogen"], NITROGEN_ORDER)
    with filter_columns[1]:
        nitrogen = _single_choice("Nitrogen", nitrogens, _state_key(prefix, "nitrogen"))
    scoped = frame.loc[frame["Water"].eq(water) & frame["Nitrogen"].eq(nitrogen)]
    with st.container():
        dates = _date_controls(scoped, variable, frame, prefix)
    if not dates:
        st.info("Select at least one date with valid measurements.")
        return

    presentation = "Single date"
    if len(dates) > 1:
        presentation = _single_choice(
            "Date display",
            ["Period summary", "Date facets", "Overlay dates"],
            _state_key(prefix, "date_display"),
            default="Period summary",
            horizontal=True,
        )
    selected = scoped.loc[scoped["Date"].isin(dates)]
    if presentation == "Period summary":
        summary, invalid_count, total_count = aggregate_period_treatment(selected, variable)
    else:
        summary, invalid_count, total_count = aggregate_daily_treatment(selected, variable)
    if summary.empty:
        st.info("No valid treatment measurements remain for this selection.")
        return

    plot_data = _prepare_summary_plot(
        summary,
        crop_frame=frame,
        dates=dates,
        presentation=presentation,
        comparison_factor=None,
        facet_factor=None,
    )
    visible_dates: list[pd.Timestamp] | None = None
    facet_columns: list[str] = []
    if presentation == "Date facets":
        visible_dates, facet_columns = _date_facets(dates, frame)
    title = (
        f"{crop} · {variable_label}<br>"
        f"{water} × {nitrogen} · {_date_summary(dates, frame)} · "
        f"{'Direct-record period summary' if presentation == 'Period summary' else 'Daily plant mean ± SD'}"
    )
    spec = _build_spec(
        plot_data,
        visible_dates=visible_dates,
        facet_columns=facet_columns,
        facet_rows=[],
        title=title,
        variable_label=variable_label,
        line_mode="lines+markers",
        color_map=color_map,
        symbol_order=symbol_order,
        dash_order=dash_order,
        summary=True,
        show_ribbon=presentation != "Overlay dates",
    )
    export = _export_table(plot_data, variable_label, identity_column)
    _render_result(
        spec,
        export,
        invalid_count=invalid_count,
        total_count=total_count,
        filename_stem=_slug(f"{crop}-{variable_label}-{water}-{nitrogen}-treatment"),
        key_prefix=prefix,
    )


def _render_individual_comparison(
    crop: str,
    frame: pd.DataFrame,
    variable: str,
    variable_label: str,
    identity_column: str,
    color_map: dict[str, str],
    symbol_order: list[str],
    dash_order: list[str],
) -> None:
    prefix = _state_key(crop, "compare", "individual_plants", variable)
    valid = frame.loc[valid_value_mask(frame, variable)]
    filter_columns = st.columns([1, 1, 1.35])
    waters = ordered_values(valid["Water"], WATER_ORDER)
    with filter_columns[0]:
        selected_waters = _multiple_choice(
            "Water", waters, _state_key(prefix, "waters"), default_all=True
        )
    valid = valid.loc[valid["Water"].isin(selected_waters)]
    nitrogens = ordered_values(valid["Nitrogen"], NITROGEN_ORDER)
    with filter_columns[1]:
        selected_nitrogens = _multiple_choice(
            "Nitrogen", nitrogens, _state_key(prefix, "nitrogens"), default_all=True
        )
    valid = valid.loc[valid["Nitrogen"].isin(selected_nitrogens)]
    dates = sorted(pd.Timestamp(value) for value in valid["Date"].unique())
    if not dates:
        st.info("No dates contain valid measurements for these treatments.")
        return
    with filter_columns[2]:
        date = _single_choice(
            "Date",
            dates,
            _state_key(prefix, "date"),
            default=dates[-1],
            format_func=lambda value: format_date(pd.Timestamp(value), frame),
        )
    valid = valid.loc[valid["Date"].eq(date)]
    labels = ordered_values(valid["Label"])
    selected_labels = _multiple_choice(
        "Plants",
        labels,
        _state_key(prefix, "labels"),
        default_all=True,
        help_text="Each plant is identified by its Label in the uploaded workbook.",
    )
    if not selected_waters or not selected_nitrogens or not selected_labels:
        st.info("Select at least one water level, nitrogen level, and plant.")
        return

    control_columns = st.columns(3)
    with control_columns[0]:
        plot_type = _single_choice(
            "Plot type",
            list(LINE_MODE_LABELS),
            _state_key(prefix, "plot_type"),
            default="Points + lines",
        )
    with control_columns[1]:
        color_label = _single_choice(
            "Color by",
            ["Plant", "Water", "Nitrogen"],
            _state_key(prefix, "color_by"),
            default="Plant",
        )
    with control_columns[2]:
        symbol_label = _single_choice(
            "Symbol by",
            [identity_column, "Water", "Nitrogen", "None"],
            _state_key(prefix, "symbol_by"),
            default=identity_column,
        )
    color_column = {"Plant": "Label", "Water": "Water", "Nitrogen": "Nitrogen"}[color_label]
    symbol_column = "PlantUnit" if symbol_label == identity_column else symbol_label

    selected = frame.loc[
        frame["Date"].eq(date)
        & frame["Water"].isin(selected_waters)
        & frame["Nitrogen"].isin(selected_nitrogens)
        & frame["Label"].isin(selected_labels)
    ]
    plants, invalid_count, total_count = aggregate_individual(selected, variable)
    if plants.empty:
        st.info("No valid plant measurements remain for this selection.")
        return
    plot_data = _prepare_individual_plot(
        plants,
        crop_frame=frame,
        color_by=color_column,
        symbol_by=symbol_column,
        date_facets=False,
        overlay_dates=False,
        single_plant=False,
    )
    title = (
        f"{crop} · {variable_label}<br>"
        f"{len(selected_labels)} plants · {format_date(pd.Timestamp(date), frame)} · "
        "Plant means after duplicate LC records are merged"
    )
    spec = _build_spec(
        plot_data,
        visible_dates=None,
        facet_columns=[],
        facet_rows=[],
        title=title,
        variable_label=variable_label,
        line_mode=LINE_MODE_LABELS[plot_type],
        color_map=color_map,
        symbol_order=symbol_order,
        dash_order=dash_order,
        summary=False,
    )
    export = _export_table(plot_data, variable_label, identity_column)
    _render_result(
        spec,
        export,
        invalid_count=invalid_count,
        total_count=total_count,
        filename_stem=_slug(f"{crop}-{variable_label}-individual-plants"),
        key_prefix=prefix,
    )


def _render_treatment_comparison(
    crop: str,
    frame: pd.DataFrame,
    variable: str,
    variable_label: str,
    identity_column: str,
    color_map: dict[str, str],
    symbol_order: list[str],
    dash_order: list[str],
    comparison_factor: str,
) -> None:
    other_factor = "Nitrogen" if comparison_factor == "Water" else "Water"
    prefix = _state_key(crop, "compare", comparison_factor, variable)
    valid = frame.loc[valid_value_mask(frame, variable)]
    filter_columns = st.columns(2)
    factor_columns = {"Water": filter_columns[0], "Nitrogen": filter_columns[1]}
    preferred = WATER_ORDER if comparison_factor == "Water" else NITROGEN_ORDER
    levels = ordered_values(valid[comparison_factor], preferred)
    with factor_columns[comparison_factor]:
        selected_levels = _multiple_choice(
            f"{comparison_factor} treatments",
            levels,
            _state_key(prefix, "levels"),
            default_all=True,
            maximum=3,
            help_text="Choose any two or three available treatment levels.",
        )
    if len(selected_levels) < 2:
        st.info(f"Select at least two {comparison_factor.lower()} treatments to compare.")
        return
    valid = valid.loc[valid[comparison_factor].isin(selected_levels)]
    other_preferred = NITROGEN_ORDER if other_factor == "Nitrogen" else WATER_ORDER
    other_levels = ordered_values(valid[other_factor], other_preferred)
    display_options = [f"Fixed {other_factor}", f"Facet by {other_factor}"]
    with factor_columns[other_factor]:
        other_display = _single_choice(
            f"{other_factor} display",
            display_options,
            _state_key(prefix, "other_display"),
            default=display_options[0],
            horizontal=True,
        )
    facet_factor: str | None = None
    if other_display.startswith("Fixed"):
        with factor_columns[other_factor]:
            fixed_level = _single_choice(
                other_factor,
                other_levels,
                _state_key(prefix, "fixed_other"),
            )
        selected_other_levels = [fixed_level]
    else:
        selected_other_levels = other_levels
        facet_factor = other_factor

    scoped = frame.loc[
        frame[comparison_factor].isin(selected_levels)
        & frame[other_factor].isin(selected_other_levels)
    ]
    with st.container():
        dates = _date_controls(scoped, variable, frame, prefix)
    if not dates:
        st.info("Select at least one date with valid measurements.")
        return
    presentation = "Single date"
    if len(dates) > 1:
        presentation = _single_choice(
            "Date display",
            ["Period summary", "Date facets", "Overlay dates"],
            _state_key(prefix, "date_display"),
            default="Period summary",
            horizontal=True,
        )

    selected = scoped.loc[scoped["Date"].isin(dates)]
    if presentation == "Period summary":
        summary, invalid_count, total_count = aggregate_period_treatment(selected, variable)
    else:
        summary, invalid_count, total_count = aggregate_daily_treatment(selected, variable)
    if summary.empty:
        st.info("No valid treatment measurements remain for this selection.")
        return

    plot_data = _prepare_summary_plot(
        summary,
        crop_frame=frame,
        dates=dates,
        presentation=presentation,
        comparison_factor=comparison_factor,
        facet_factor=facet_factor,
    )
    visible_dates: list[pd.Timestamp] | None = None
    facet_columns: list[str] = []
    if presentation == "Date facets":
        visible_dates, facet_columns = _date_facets(dates, frame)
    facet_rows = (
        [f"{facet_factor}: {value}" for value in selected_other_levels] if facet_factor else []
    )
    other_context = (
        f"{other_factor} facets" if facet_factor else f"{other_factor} {selected_other_levels[0]}"
    )
    calculation = (
        "Direct-record period summary" if presentation == "Period summary" else "Daily plant mean ± SD"
    )
    title = (
        f"{crop} · {variable_label}<br>"
        f"{comparison_factor} comparison · {other_context} · {_date_summary(dates, frame)} · {calculation}"
    )
    spec = _build_spec(
        plot_data,
        visible_dates=visible_dates,
        facet_columns=facet_columns,
        facet_rows=facet_rows,
        title=title,
        variable_label=variable_label,
        line_mode="lines+markers",
        color_map=color_map,
        symbol_order=symbol_order,
        dash_order=dash_order,
        summary=True,
        show_ribbon=presentation != "Overlay dates",
    )
    export = _export_table(plot_data, variable_label, identity_column)
    _render_result(
        spec,
        export,
        invalid_count=invalid_count,
        total_count=total_count,
        filename_stem=_slug(f"{crop}-{variable_label}-{comparison_factor}-comparison"),
        key_prefix=prefix,
    )


def render_crop_panel(crop: str, path: Path, function: str) -> None:
    """Render one isolated crop panel, including its own load errors and state."""

    st.markdown(f"### {crop} light-response curves")
    if not path.exists():
        st.error(f"Data file not found: {path.name}")
        return
    try:
        modified_time_ns, file_size = file_signature(path)
        frame = load_workbook(str(path.resolve()), modified_time_ns, file_size)
    except (OSError, DataValidationError) as exc:
        st.error(str(exc))
        return

    variables = available_variables(frame)
    if not variables:
        st.error(
            f"{path.name} does not contain valid numeric data in any supported response variable."
        )
        return
    labels_by_column = {column: label for label, column in variables}
    columns = list(labels_by_column)
    default_variable = "Y(II)_calc" if "Y(II)_calc" in columns else columns[0]
    variable_key = _state_key(crop, "variable")
    if st.session_state.get(variable_key) not in columns:
        st.session_state[variable_key] = default_variable

    st.caption(
        f"{len(frame):,} records · {frame['Label'].nunique():,} plants · "
        f"{frame['Date'].nunique():,} dates · source: {path.name}"
    )
    st.markdown("#### Response variable")
    variable = st.radio(
        "Response variable",
        columns,
        key=variable_key,
        format_func=labels_by_column.get,
        horizontal=True,
        label_visibility="collapsed",
    )
    variable_label = VARIABLE_LABELS[variable]
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

    color_map, symbol_order, dash_order = _style_maps(frame)
    identity_column = str(frame["PlantUnitColumn"].iloc[0])

    if function == "Explore Data":
        plot_kind = _single_choice(
            "Explore view",
            ["Single plant", "Treatment curve"],
            _state_key(crop, "explore_kind"),
            default="Single plant",
            horizontal=True,
        )
        st.markdown("#### Filters")
        if plot_kind == "Single plant":
            _render_single_plant(
                crop,
                frame,
                variable,
                variable_label,
                identity_column,
                color_map,
                symbol_order,
                dash_order,
            )
        else:
            _render_treatment_curve(
                crop,
                frame,
                variable,
                variable_label,
                identity_column,
                color_map,
                symbol_order,
                dash_order,
            )
    else:
        plot_kind = _single_choice(
            "Compare view",
            ["Individual plants", "Water treatments", "Nitrogen treatments"],
            _state_key(crop, "compare_kind"),
            default="Individual plants",
            horizontal=True,
        )
        st.markdown("#### Filters")
        if plot_kind == "Individual plants":
            _render_individual_comparison(
                crop,
                frame,
                variable,
                variable_label,
                identity_column,
                color_map,
                symbol_order,
                dash_order,
            )
        elif plot_kind == "Water treatments":
            _render_treatment_comparison(
                crop,
                frame,
                variable,
                variable_label,
                identity_column,
                color_map,
                symbol_order,
                dash_order,
                "Water",
            )
        else:
            _render_treatment_comparison(
                crop,
                frame,
                variable,
                variable_label,
                identity_column,
                color_map,
                symbol_order,
                dash_order,
                "Nitrogen",
            )
