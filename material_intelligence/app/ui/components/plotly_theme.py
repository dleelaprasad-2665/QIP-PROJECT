"""Shared Plotly figure styling — BuildShield orange / black palette."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from material_intelligence.app.ui.state import get_app_state

ACCENT = "#e8940a"
ACCENT_ALT = "#f5a623"
MUTED_SERIES = ["#e8940a", "#1a1a1a", "#c47808", "#1f8f63", "#6b655c", "#c53d3d"]


def _layout(**kwargs: Any) -> dict[str, Any]:
    mode = get_app_state().theme_mode
    if mode == "light":
        paper = "#ffffff"
        plot = "#ffffff"
        font = "#111111"
        grid = "#e4e1db"
    else:
        paper = "#141414"
        plot = "#141414"
        font = "#f5f3ef"
        grid = "#2e2e2e"
    base = dict(
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        font=dict(color=font, family="Barlow, sans-serif", size=12),
        title=dict(font=dict(family="Oswald, sans-serif", size=14), x=0.02, xanchor="left"),
        margin=dict(l=44, r=18, t=52, b=42),
        height=290,
        xaxis=dict(gridcolor=grid, zerolinecolor=grid, showline=False),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid, showline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )
    base.update(kwargs)
    return base


def line_chart(x: list, y: list, *, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=dict(color=ACCENT, width=2.5),
                marker=dict(size=6, color=ACCENT),
                fill="tozeroy",
                fillcolor="rgba(232,148,10,0.12)",
            )
        ]
    )
    fig.update_layout(**_layout(title=title))
    return fig


def bar_chart(x: list, y: list, *, title: str, orientation: str = "v") -> go.Figure:
    if orientation == "h":
        data = go.Bar(x=y, y=x, orientation="h", marker_color=ACCENT, marker_line_width=0)
    else:
        data = go.Bar(x=x, y=y, marker_color=ACCENT, marker_line_width=0)
    fig = go.Figure(data=[data])
    fig.update_layout(**_layout(title=title))
    return fig


def pie_chart(labels: list, values: list, *, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.52,
                marker=dict(colors=MUTED_SERIES[: max(len(labels), 1)]),
                textinfo="percent+label",
                textposition="outside",
            )
        ]
    )
    fig.update_layout(**_layout(title=title, showlegend=False))
    return fig


def grouped_outcomes(sites: list, statuses: list, counts: list, *, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=[f"{s}" for s in sites],
                y=counts,
                text=statuses,
                marker_color=ACCENT_ALT,
                hovertemplate="%{x}<br>%{text}: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(**_layout(title=title))
    return fig


def confusion_heatmap(matrix: list[list[int]], class_names: list[str], *, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=class_names,
            y=class_names,
            colorscale=[[0, "#f5f3ef"], [0.5, "#f0c46a"], [1, "#e8940a"]],
        )
    )
    fig.update_layout(**_layout(title=title, height=320))
    return fig
