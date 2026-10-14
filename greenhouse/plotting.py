"""Interactive and static renderers backed by the same prepared plot data."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from math import ceil

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from greenhouse.data import VARIABLE_UNITS


WATER_COLORS = {"WW": "#2563EB", "DT": "#DC2626", "RW": "#16A34A"}
NITROGEN_COLORS = {"N0": "#EA8C00", "N1": "#2563EB", "N2": "#16A34A"}
GENERAL_COLORS = (
    "#0F766E",
    "#C2410C",
    "#2563EB",
    "#7C3AED",
    "#BE123C",
    "#4D7C0F",
    "#0369A1",
    "#A16207",
    "#6D28D9",
    "#0E7490",
    "#B91C1C",
    "#3F6212",
    "#1D4ED8",
    "#9F1239",
    "#115E59",
    "#92400E",
)
PLOTLY_SYMBOLS = (
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "triangle-up",
    "triangle-down",
    "pentagon",
    "hexagon",
    "star",
)
MATPLOTLIB_SYMBOLS = ("o", "s", "D", "P", "X", "^", "v", "p", "h", "*")
PLOTLY_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
MATPLOTLIB_DASHES = ("-", "--", ":", "-.", (0, (7, 3)), (0, (5, 2, 1, 2)))


@dataclass
class PlotAnnotation:
    """One annotation box with equivalent interactive and static text."""

    html_text: str
    text: str
    facet_column: str = ""
    facet_row: str = ""


@dataclass
class PlotSpec:
    """A renderer-neutral description of the current chart."""

    all_data: pd.DataFrame
    visible_data: pd.DataFrame
    title: str
    y_label: str
    value_column: str
    error_column: str | None
    line_mode: str
    color_map: dict[str, str]
    symbol_order: list[str]
    dash_order: list[str]
    facet_rows: list[str]
    facet_columns: list[str]
    show_ribbon: bool = False
    x_column: str = "PAR"
    x_label: str = "PAR (µmol m⁻² s⁻¹)"
    annotations: list[PlotAnnotation] = field(default_factory=list)
    include_y_zero: bool = False


def stable_color_map(values: list[str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Create a deterministic category palette with optional treatment colors."""

    result = dict(overrides or {})
    unassigned = [value for value in values if value not in result]
    for index, value in enumerate(unassigned):
        result[value] = GENERAL_COLORS[index % len(GENERAL_COLORS)]
    return result


def axis_label(variable_label: str) -> str:
    """Keep selector labels short and put physical units on the response axis."""

    unit = VARIABLE_UNITS.get(variable_label)
    return f"{variable_label} ({unit})" if unit else variable_label


def _rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    red, green, blue = (int(color[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _category_style(value: object, order: list[str], choices: tuple[str, ...]) -> str:
    key = str(value)
    try:
        index = order.index(key)
    except ValueError:
        index = 0
    return choices[index % len(choices)]


def _axis_range(spec: PlotSpec) -> tuple[list, list[float]]:
    frame = spec.all_data
    y_values = pd.to_numeric(frame[spec.value_column], errors="coerce").to_numpy(dtype=float)
    y_values = y_values[np.isfinite(y_values)]

    if spec.error_column and spec.error_column in frame:
        errors = pd.to_numeric(frame[spec.error_column], errors="coerce").to_numpy(dtype=float)
        centers = pd.to_numeric(frame[spec.value_column], errors="coerce").to_numpy(dtype=float)
        error_mask = np.isfinite(errors) & np.isfinite(centers)
        if error_mask.any():
            y_values = np.concatenate(
                [y_values, centers[error_mask] - errors[error_mask], centers[error_mask] + errors[error_mask]]
            )

    def padded(values: np.ndarray) -> list[float]:
        minimum = float(values.min())
        maximum = float(values.max())
        span = maximum - minimum
        padding = span * 0.05 if span else max(abs(maximum) * 0.05, 0.05)
        return [minimum - padding, maximum + padding]

    if spec.x_column == "Date":
        dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
        minimum, maximum = dates.min(), dates.max()
        padding = (maximum - minimum) * 0.05 if maximum > minimum else pd.Timedelta(days=1)
        x_range = [minimum - padding, maximum + padding]
    else:
        x_values = pd.to_numeric(frame[spec.x_column], errors="coerce").to_numpy(dtype=float)
        x_range = padded(x_values[np.isfinite(x_values)])
    y_range = padded(y_values)
    if spec.include_y_zero:
        y_range[0] = min(0, y_range[0])
        y_range[1] = max(0, y_range[1])
    return x_range, y_range


def _facet_values(frame: pd.DataFrame, column: str, configured: list[str]) -> list[str]:
    if configured:
        return configured
    if column not in frame:
        return [""]
    values = [str(value) for value in frame[column].dropna().unique()]
    return values or [""]


def _legend_duplicates_facet_titles(data: pd.DataFrame, columns: list[str]) -> bool:
    """Return whether every legend label merely repeats its date facet title."""

    if not any(columns) or not {"FacetColumn", "LegendLabel"}.issubset(data.columns):
        return False
    facets = data.loc[data["FacetColumn"].astype(str).ne("")]
    return bool(
        not facets.empty
        and facets["LegendLabel"].astype(str).eq(facets["FacetColumn"].astype(str)).all()
    )


def create_plotly_figure(spec: PlotSpec) -> go.Figure:
    """Render a Plotly figure with shared ranges and optional SD ribbons."""

    data = spec.visible_data
    rows = _facet_values(data, "FacetRow", spec.facet_rows)
    columns = _facet_values(data, "FacetColumn", spec.facet_columns)
    has_column_facets = any(columns)
    show_legend = not _legend_duplicates_facet_titles(data, columns)
    facet_title_space = 50 if has_column_facets else 0
    legend_count = data["LegendLabel"].astype(str).nunique() if show_legend else 0
    legend_rows = ceil(legend_count / 6) if legend_count else 0
    legend_space = 80 + (32 * (legend_rows - 1)) if legend_rows else 0
    shared_axis_label_space = 12 if has_column_facets else 0
    base_height = max(520, 360 * len(rows))
    plot_height = max(1, base_height - 175)
    # Keep the legend below long, two-line titles while retaining a clear gap
    # above the date-facet headings.
    title_legend_clearance = 36 if has_column_facets and legend_rows else 0
    legend_y = 1.02 + (max(0, facet_title_space - title_legend_clearance) / plot_height)
    subplot_titles = [column if row_index == 0 else "" for row_index, _ in enumerate(rows) for column in columns]
    figure = make_subplots(
        rows=len(rows),
        cols=len(columns),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=min(0.035, 0.12 / max(1, len(columns) - 1)),
        vertical_spacing=0.11,
        subplot_titles=subplot_titles,
    )
    x_range, y_range = _axis_range(spec)
    seen_legends: set[str] = set()

    for row_index, row_value in enumerate(rows, start=1):
        for column_index, column_value in enumerate(columns, start=1):
            panel = data
            if "FacetRow" in panel:
                panel = panel.loc[panel["FacetRow"].astype(str).eq(row_value)]
            if "FacetColumn" in panel:
                panel = panel.loc[panel["FacetColumn"].astype(str).eq(column_value)]
            if panel.empty:
                figure.update_xaxes(range=x_range, row=row_index, col=column_index)
                figure.update_yaxes(range=y_range, row=row_index, col=column_index)
                figure.add_annotation(
                    text="No data",
                    x=sum(x_range) / 2,
                    y=sum(y_range) / 2,
                    showarrow=False,
                    font={"size": 16, "color": "#64748B"},
                    row=row_index,
                    col=column_index,
                )
                continue

            for _, series in panel.groupby("SeriesKey", observed=True, sort=False):
                series = series.sort_values(spec.x_column, kind="stable")
                legend_label = str(series["LegendLabel"].iloc[0])
                color_key = str(series["ColorKey"].iloc[0])
                color = spec.color_map.get(color_key, GENERAL_COLORS[0])
                symbol_key = str(series["SymbolKey"].iloc[0])
                dash_key = str(series["DashKey"].iloc[0])
                symbol = _category_style(symbol_key, spec.symbol_order, PLOTLY_SYMBOLS)
                dash = _category_style(dash_key, spec.dash_order, PLOTLY_DASHES)

                if spec.show_ribbon and spec.error_column:
                    errors = pd.to_numeric(series[spec.error_column], errors="coerce")
                    ribbon = series.loc[errors.notna() & np.isfinite(errors)].copy()
                    if not ribbon.empty:
                        upper = ribbon[spec.value_column] + ribbon[spec.error_column]
                        lower = ribbon[spec.value_column] - ribbon[spec.error_column]
                        figure.add_trace(
                            go.Scatter(
                                x=pd.concat([ribbon[spec.x_column], ribbon[spec.x_column].iloc[::-1]]),
                                y=pd.concat([upper, lower.iloc[::-1]]),
                                mode="lines",
                                fill="toself",
                                fillcolor=_rgba(color, 0.16),
                                line={"width": 0},
                                hoverinfo="skip",
                                showlegend=False,
                            ),
                            row=row_index,
                            col=column_index,
                        )

                figure.add_trace(
                    go.Scatter(
                        x=series[spec.x_column],
                        y=series[spec.value_column],
                        mode=str(series["LineMode"].iloc[0]) if "LineMode" in series else spec.line_mode,
                        name=legend_label,
                        legendgroup=legend_label,
                        showlegend=show_legend and legend_label not in seen_legends,
                        line={"color": color, "width": 2.2, "dash": dash},
                        marker={
                            "color": color,
                            "symbol": symbol,
                            "size": 7.5,
                            "line": {"color": "white", "width": 0.7},
                            "opacity": float(series["MarkerOpacity"].iloc[0]) if "MarkerOpacity" in series else 1,
                        },
                        text=series["HoverText"],
                        hovertemplate="%{text}<extra></extra>",
                        connectgaps=False,
                    ),
                    row=row_index,
                    col=column_index,
                )
                seen_legends.add(legend_label)

            figure.update_xaxes(range=x_range, row=row_index, col=column_index)
            figure.update_yaxes(range=y_range, row=row_index, col=column_index)

        if row_value:
            figure.add_annotation(
                text=row_value,
                xref="paper",
                yref="paper",
                x=1.015,
                y=1 - ((row_index - 0.5) / len(rows)),
                textangle=90,
                showarrow=False,
                font={"size": 16, "color": "#475569"},
            )

    figure.update_layout(
        title={
            "text": spec.title,
            "x": 0.015,
            "xanchor": "left",
            "font": {"size": 23},
        },
        template="plotly_white",
        height=base_height + facet_title_space + legend_space + shared_axis_label_space,
        margin={
            "l": 78,
            "r": 54 if any(rows) else 28,
            "t": 105 + facet_title_space + legend_space,
            "b": 70 + shared_axis_label_space,
        },
        hovermode="closest",
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": legend_y,
            "xanchor": "right",
            "x": 1,
            "title": None,
            "font": {"size": 15},
        },
        font={"family": "Geist, Segoe UI, sans-serif", "size": 16, "color": "#1F2937"},
        hoverlabel={
            "font": {"family": "Geist, Segoe UI, sans-serif", "size": 16},
            "bgcolor": "#FFFFFF",
            "bordercolor": "#CBD5E1",
        },
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    figure.update_xaxes(
        title_text=None if has_column_facets else spec.x_label,
        showgrid=True,
        gridcolor="#E8ECE8",
        zeroline=False,
        tickfont={"size": 15},
        title_font={"size": 17},
    )
    figure.update_yaxes(
        title_text=None if has_column_facets else spec.y_label,
        showgrid=True,
        gridcolor="#E8ECE8",
        zeroline=False,
        tickfont={"size": 15},
        title_font={"size": 17},
    )
    figure.update_annotations(font={"size": 16, "color": "#334155"})
    if has_column_facets:
        figure.add_annotation(
            text=spec.x_label,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0,
            yshift=-64,
            showarrow=False,
            font={"size": 17, "color": "#334155"},
        )
        figure.add_annotation(
            text=spec.y_label,
            xref="paper",
            yref="paper",
            x=0,
            xshift=-56,
            y=0.5,
            textangle=-90,
            showarrow=False,
            font={"size": 17, "color": "#334155"},
        )
    for row_index, row_value in enumerate(rows, start=1):
        for column_index, column_value in enumerate(columns, start=1):
            axis_index = (row_index - 1) * len(columns) + column_index
            axis_suffix = "" if axis_index == 1 else str(axis_index)
            for annotation in spec.annotations:
                if annotation.facet_row != row_value or annotation.facet_column != column_value:
                    continue
                figure.add_annotation(
                    text=annotation.html_text,
                    xref=f"x{axis_suffix} domain", yref=f"y{axis_suffix} domain",
                    x=0.98, y=0.04, xanchor="right", yanchor="bottom",
                    showarrow=False, align="left",
                    bgcolor="rgba(255,255,255,0.75)",
                    bordercolor="rgba(100,116,139,0.4)", borderwidth=1, borderpad=8,
                    font={"size": 14, "color": "#1F2937"},
                )
    return figure


def _matplotlib_mode(line_mode: str) -> tuple[str, bool]:
    if line_mode == "markers":
        return "None", True
    if line_mode == "lines":
        return "-", False
    return "-", True


def create_static_export(spec: PlotSpec, file_format: str) -> bytes:
    """Render the visible page as a 300 DPI PNG or vector PDF."""

    data = spec.visible_data
    rows = _facet_values(data, "FacetRow", spec.facet_rows)
    columns = _facet_values(data, "FacetColumn", spec.facet_columns)
    has_column_facets = any(columns)
    show_legend = not _legend_duplicates_facet_titles(data, columns)
    figure, axes = plt.subplots(
        len(rows),
        len(columns),
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=(max(7.2, 4.8 * len(columns)), max(4.8, 3.8 * len(rows))),
        gridspec_kw={"wspace": 0.08},
        constrained_layout=True,
    )
    x_range, y_range = _axis_range(spec)
    legend_handles: dict[str, object] = {}

    for row_index, row_value in enumerate(rows):
        for column_index, column_value in enumerate(columns):
            axis = axes[row_index][column_index]
            panel = data
            if "FacetRow" in panel:
                panel = panel.loc[panel["FacetRow"].astype(str).eq(row_value)]
            if "FacetColumn" in panel:
                panel = panel.loc[panel["FacetColumn"].astype(str).eq(column_value)]

            if panel.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#64748B",
                )

            for _, series in panel.groupby("SeriesKey", observed=True, sort=False):
                series = series.sort_values(spec.x_column, kind="stable")
                legend_label = str(series["LegendLabel"].iloc[0])
                color_key = str(series["ColorKey"].iloc[0])
                color = spec.color_map.get(color_key, GENERAL_COLORS[0])
                symbol_key = str(series["SymbolKey"].iloc[0])
                dash_key = str(series["DashKey"].iloc[0])
                marker = _category_style(symbol_key, spec.symbol_order, MATPLOTLIB_SYMBOLS)
                dash_style = _category_style(dash_key, spec.dash_order, MATPLOTLIB_DASHES)
                line_style, show_markers = _matplotlib_mode(
                    str(series["LineMode"].iloc[0]) if "LineMode" in series else spec.line_mode
                )
                marker_opacity = float(series["MarkerOpacity"].iloc[0]) if "MarkerOpacity" in series else 1

                if spec.show_ribbon and spec.error_column:
                    errors = pd.to_numeric(series[spec.error_column], errors="coerce")
                    ribbon = series.loc[errors.notna() & np.isfinite(errors)]
                    if not ribbon.empty:
                        axis.fill_between(
                            ribbon[spec.x_column].to_numpy(),
                            (ribbon[spec.value_column] - ribbon[spec.error_column]).to_numpy(dtype=float),
                            (ribbon[spec.value_column] + ribbon[spec.error_column]).to_numpy(dtype=float),
                            color=color,
                            alpha=0.16,
                            linewidth=0,
                        )

                effective_line = dash_style if line_style != "None" else "None"
                (handle,) = axis.plot(
                    series[spec.x_column],
                    series[spec.value_column],
                    color=color,
                    linestyle=effective_line,
                    linewidth=1.8,
                    marker=marker if show_markers else None,
                    markersize=5.2,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    markerfacecolor=matplotlib.colors.to_rgba(color, marker_opacity),
                    label=legend_label,
                )
                legend_handles.setdefault(legend_label, handle)

            axis.set_xlim(x_range)
            if spec.x_column == "Date":
                locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
                axis.xaxis.set_major_locator(locator)
                axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            axis.set_ylim(y_range)
            axis.grid(True, color="#E8ECE8", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="both", labelsize=11)
            if row_index == len(rows) - 1 and not has_column_facets:
                axis.set_xlabel(spec.x_label, fontsize=12.5)
            if column_index == 0 and not has_column_facets:
                axis.set_ylabel(spec.y_label, fontsize=12.5)
            if column_value:
                axis.set_title(column_value, fontsize=13, fontweight="bold")
            if row_value and column_index == len(columns) - 1:
                axis.text(
                    1.03,
                    0.5,
                    row_value,
                    transform=axis.transAxes,
                    rotation=-90,
                    va="center",
                    ha="left",
                    fontsize=11.5,
                    color="#475569",
                )
            for annotation in spec.annotations:
                if annotation.facet_row != row_value or annotation.facet_column != column_value:
                    continue
                axis.text(
                    0.98, 0.04, annotation.text, transform=axis.transAxes,
                    ha="right", va="bottom", fontsize=9.5, linespacing=1.3,
                    color="#1F2937", zorder=5,
                    bbox={
                        "boxstyle": "round,pad=0.45",
                        "facecolor": (1, 1, 1, 0.75),
                        "edgecolor": (100 / 255, 116 / 255, 139 / 255, 0.4),
                    },
                )

    figure.suptitle(
        spec.title.replace("<br>", "\n"),
        x=0.01,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    if has_column_facets:
        figure.supxlabel(spec.x_label, fontsize=12.5)
        figure.supylabel(spec.y_label, fontsize=12.5)
    if legend_handles and show_legend:
        figure.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="outside upper right",
            ncols=min(4, len(legend_handles)),
            frameon=False,
            fontsize=10.5,
        )
    output = BytesIO()
    figure.savefig(output, format=file_format, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output.getvalue()
