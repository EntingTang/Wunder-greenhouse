"""Treatment distributions with shared statistics for interactive/static plots."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from greenhouse.plotting import PlotSpec


def distribution_statistics(frame: pd.DataFrame, value_column: str = "Value") -> pd.DataFrame:
    """Calculate linear quartiles and 1.5-IQR whisker endpoints for n >= 3."""

    records = []
    for date, group in frame.groupby("Date", sort=True):
        values = pd.to_numeric(group[value_column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        record = {
            "Date": date, "n": len(values), "Mean": float(values.mean()),
            "Median": float(np.median(values)), "Q1": np.nan, "Q3": np.nan,
            "Whisker lower": np.nan, "Whisker upper": np.nan, "Box shown": len(values) >= 3,
        }
        if record["Box shown"]:
            q1, q3 = np.quantile(values, [0.25, 0.75], method="linear")
            iqr = q3 - q1
            within = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
            record.update({
                "Q1": float(q1), "Q3": float(q3),
                "Whisker lower": float(min(q1, within.min())),
                "Whisker upper": float(max(q3, within.max())),
            })
        records.append(record)
    return pd.DataFrame(records)


def _plot_groups(spec: PlotSpec) -> list[dict]:
    stats = distribution_statistics(spec.visible_data, spec.value_column).set_index("Date")
    groups = []
    for index, (date, group) in enumerate(spec.visible_data.groupby("Date", sort=True)):
        group = group.sort_values("Label", kind="stable")
        color_key = str(group["ColorKey"].iloc[0])
        offsets = np.linspace(-0.16, 0.16, len(group)) if len(group) > 1 else np.array([-0.12])
        groups.append({
            "position": index, "points_x": index + offsets,
            "values": group[spec.value_column].to_numpy(dtype=float),
            "hover": group["HoverText"].tolist(), "label": str(group["DateDisplay"].iloc[0]),
            "color": spec.color_map[color_key], "stats": stats.loc[date].to_dict(),
        })
    return groups


def _y_range(groups: list[dict]) -> tuple[float, float]:
    values = np.concatenate([group["values"] for group in groups])
    lower, upper = float(values.min()), float(values.max())
    padding = (upper - lower) * 0.05 if upper > lower else max(abs(upper) * 0.05, 0.05)
    return lower - padding, upper + padding


def create_box_figure(spec: PlotSpec) -> go.Figure:
    groups = _plot_groups(spec)
    figure = go.Figure()
    first_box = True
    for group in groups:
        stats, position, color = group["stats"], group["position"], group["color"]
        if not stats["Box shown"]:
            continue
        red, green, blue = (round(channel * 255) for channel in mcolors.to_rgb(color))
        figure.add_trace(go.Box(
            x=[position], q1=[stats["Q1"]], median=[stats["Median"]], q3=[stats["Q3"]],
            lowerfence=[stats["Whisker lower"]], upperfence=[stats["Whisker upper"]],
            width=0.52, boxpoints=False, boxmean=False,
            fillcolor=f"rgba({red},{green},{blue},0.25)",
            line={"color": color, "width": 1.4},
            name="Box (Q1–Q3)", legendgroup="box", showlegend=first_box,
            hovertemplate=(
                f"{group['label']}<br>n = {stats['n']}<br>"
                f"Median: {stats['Median']:.6g}<br>Q1: {stats['Q1']:.6g}<br>"
                f"Q3: {stats['Q3']:.6g}<br>Whiskers: {stats['Whisker lower']:.6g} "
                f"to {stats['Whisker upper']:.6g}<extra></extra>"
            ),
        ))
        first_box = False
        figure.add_trace(go.Scatter(
            x=[position - 0.26, position + 0.26], y=[stats["Median"]] * 2,
            mode="lines", line={"color": color, "width": 3.5},
            showlegend=False, hoverinfo="skip",
        ))
    figure.add_trace(go.Scatter(
        x=np.concatenate([group["points_x"] for group in groups]),
        y=np.concatenate([group["values"] for group in groups]),
        text=[hover for group in groups for hover in group["hover"]],
        mode="markers", name="Individual values",
        marker={"color": "white", "size": 8, "line": {"color": "#334155", "width": 1.2}},
        hovertemplate="%{text}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=[group["position"] for group in groups],
        y=[group["stats"]["Mean"] for group in groups], mode="markers", name="Mean",
        marker={"color": "black", "symbol": "diamond", "size": 11},
        text=[f"{group['label']}<br>Mean: {group['stats']['Mean']:.6g}<br>n = {group['stats']['n']}"
              for group in groups],
        hovertemplate="%{text}<extra></extra>",
    ))
    figure.update_layout(
        title={"text": spec.title, "x": 0.015, "xanchor": "left", "font": {"size": 23}},
        template="plotly_white", height=640,
        margin={"l": 78, "r": 28, "t": 165, "b": 105},
        font={"family": "Geist, Segoe UI, sans-serif", "size": 16, "color": "#1F2937"},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.04, "yanchor": "bottom"},
        hovermode="closest", plot_bgcolor="white", paper_bgcolor="white",
    )
    figure.update_xaxes(
        title_text=spec.x_label, range=[-0.5, len(groups) - 0.5],
        tickvals=[group["position"] for group in groups],
        ticktext=[group["label"].replace(" (", "<br>(") for group in groups],
        tickfont={"size": 12}, showgrid=False, zeroline=False,
    )
    figure.update_yaxes(title_text=spec.y_label, range=_y_range(groups), gridcolor="#E8ECE8", zeroline=False)
    return figure


def create_box_export(spec: PlotSpec, file_format: str) -> bytes:
    groups = _plot_groups(spec)
    figure, axis = plt.subplots(figsize=(max(7.2, 0.95 * len(groups)), 5.4), constrained_layout=True)
    has_box = False
    for group in groups:
        stats, position, color = group["stats"], group["position"], group["color"]
        if stats["Box shown"]:
            axis.bxp([{
                "q1": stats["Q1"], "med": stats["Median"], "q3": stats["Q3"],
                "whislo": stats["Whisker lower"], "whishi": stats["Whisker upper"],
            }], positions=[position], widths=0.52, patch_artist=True, showfliers=False,
                manage_ticks=False, boxprops={"facecolor": mcolors.to_rgba(color, 0.25), "edgecolor": color},
                whiskerprops={"color": color, "linewidth": 1.2},
                capprops={"color": color, "linewidth": 1.2},
                medianprops={"color": color, "linewidth": 2.8})
            has_box = True
        axis.scatter(group["points_x"], group["values"], s=28, facecolors="white", edgecolors="#334155", zorder=3)
        axis.scatter([position], [stats["Mean"]], s=42, marker="D", color="black", zorder=4)
    axis.set_xticks([group["position"] for group in groups], [group["label"].replace(" (", "\n(") for group in groups])
    axis.set_xlim(-0.5, len(groups) - 0.5)
    axis.set_ylim(*_y_range(groups))
    axis.set_xlabel(spec.x_label, fontsize=12.5)
    axis.set_ylabel(spec.y_label, fontsize=12.5)
    axis.grid(axis="y", color="#E8ECE8", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=9)
    axis.tick_params(axis="y", labelsize=11)
    handles = []
    if has_box:
        color = groups[0]["color"]
        handles.append(Patch(facecolor=mcolors.to_rgba(color, 0.25), edgecolor=color, label="Box (Q1–Q3)"))
    handles.extend([
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor="#334155", label="Individual values"),
        Line2D([], [], linestyle="none", marker="D", color="black", label="Mean"),
    ])
    figure.suptitle(spec.title.replace("<br>", "\n"), x=0.01, ha="left", fontsize=16, fontweight="bold")
    figure.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False)
    output = BytesIO()
    figure.savefig(output, format=file_format, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output.getvalue()
