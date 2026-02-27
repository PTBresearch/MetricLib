import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Optional, Union, Any

COLORS = [
    "#FF6347",
    "#008080",
    "#1E90FF",
    "#FFD700",
]


def add_bar(
    name: str,
    labels: List[str],
    figure: go.Figure,
    histogram: Dict[str, Union[int, float]],
    filtered_histogram: Optional[Dict[str, Union[int, float]]] = None,
    xaxis_title: str = "",
    yaxis_title: str = "",
) -> go.Figure:
    """
    Append bar traces (counts and percentages) to a Plotly figure, with an
    optional background for the unfiltered population.

    When a `filtered_histogram` is provided, two traces are added for each mode:
    - Absolute mode: gray "(all)" bars behind the colored filtered bars.
    - Relative mode: percent-normalized "(all) %" and filtered "%" bars.

    A dropdown menu is added to toggle between "Total Values" (counts) and
    "Relative" (percentages). The y-axis title switches accordingly.

    Parameters:
        name (str): Series name used in trace labels and hover.
        labels (List[str]): Ordered category labels for the x-axis.
        figure (plotly.graph_objects.Figure): Figure to augment; returned figure
            contains prior data plus the newly added traces and layout updates.
        histogram (Dict[str, int | float]): Counts per label for the full
            population. Used for absolute bars and to compute "(all) %" in
            relative mode.
        filtered_histogram (Optional[Dict[str, int | float]]): Counts per label
            for the filtered subset. If None, only a single series (all) is
            added.
        xaxis_title (str): X-axis title applied to the resulting figure.
        yaxis_title (str): Initial y-axis title; toggling modes will update it
            to "Counts" or "Percentage".

    Returns:
        plotly.graph_objects.Figure: Figure with appended bar traces and an
        interactive toggle between absolute and relative views.
    """
    color_index = int(len(figure.data) / 2)
    base_color = COLORS[color_index]

    bars = []

    if filtered_histogram:
        bars.append(
            go.Bar(
                x=labels,
                y=[histogram.get(label, 0) for label in labels],
                name=f"{name} (all)",
                marker_color="gray",
                opacity=0.3,
                offsetgroup=color_index,
                showlegend=False,
                visible=True,
            )
        )

        bars.append(
            go.Bar(
                x=labels,
                y=[filtered_histogram.get(label, 0) for label in labels],
                name=name,
                marker_color=base_color,
                opacity=1,
                offsetgroup=color_index,
                showlegend=False,
                visible=True,
            )
        )

        bars.append(
            go.Bar(
                x=labels,
                y=[
                    histogram.get(label, 0) / np.sum(list(histogram.values()))
                    for label in labels
                ],
                name=f"{name} (all) %",
                marker_color="gray",
                opacity=0.3,
                offsetgroup=color_index,
                showlegend=False,
                visible=False,
            )
        )
        bars.append(
            go.Bar(
                x=labels,
                y=[
                    filtered_histogram.get(label, 0)
                    / np.sum(list(filtered_histogram.values()))
                    for label in labels
                ],
                name=f"{name} %",
                marker_color=base_color,
                opacity=1,
                offsetgroup=color_index,
                showlegend=False,
                visible=False,
            )
        )
    else:
        bars.append(
            go.Bar(
                x=labels,
                y=[histogram.get(label, 0) for label in labels],
                name=name,
                marker_color=base_color,
                opacity=1,
                offsetgroup=color_index,
                showlegend=False,
                visible=True,
            )
        )
        bars.append(
            go.Bar(
                x=labels,
                y=[
                    histogram.get(label, 0) / np.sum(list(histogram.values()))
                    for label in labels
                ],
                name=f"{name} %",
                marker_color=base_color,
                opacity=1,
                offsetgroup=color_index,
                showlegend=False,
                visible=False,
            )
        )

    figure = go.Figure(data=list(figure.data) + bars)

    num_traces = len(figure.data)
    step = 4 if filtered_histogram else 2
    visibility_total = []
    visibility_relative = []

    for i in range(0, num_traces, step):
        if filtered_histogram:
            visibility_total += [
                True,
                True,
                False,
                False,
            ]
            visibility_relative += [
                False,
                False,
                False,
                True,
            ]
        else:
            visibility_total += [True, False]
            visibility_relative += [False, True]

    layout = go.Layout(
        barmode="group",
        xaxis=dict(
            title=xaxis_title, showgrid=False, linecolor="#A9A9A9", type="category"
        ),
        yaxis=dict(
            title=yaxis_title,
            showline=True,
            showgrid=False,
            linecolor="#A9A9A9",
        ),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        args=[
                            {"visible": visibility_total},
                            {
                                "yaxis": {
                                    "title": "Counts",
                                    "showgrid": False,
                                    "showline": True,
                                }
                            },
                        ],
                        label="Total Values",
                        method="update",
                    ),
                    dict(
                        args=[
                            {"visible": visibility_relative},
                            {
                                "yaxis": {
                                    "title": "Percentage",
                                    "showgrid": False,
                                    "showline": True,
                                }
                            },
                        ],
                        label="Relative",
                        method="update",
                    ),
                ],
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="black",
                bordercolor="gray",
                font=dict(color="white"),
            )
        ],
    )

    figure.update_layout(layout)

    return figure


def build_mosaique_figure(
    filtered_metadata, labels, index, category_field, proportion_field, name
) -> go.Figure:
    filtered_metadata = filtered_metadata[index]
    filtered_metadata["labels"] = labels[index]
    proportion_values = filtered_metadata[proportion_field].unique()
    for proportion_value in proportion_values:
        metadata = filtered_metadata
        metadata = metadata.explode(category_field)
        metadata[proportion_field] = metadata[proportion_field] == proportion_value
        class_counts = (
            metadata.groupby([category_field, proportion_field])
            .size()
            .rename("_class_counts")
            .reset_index()
        ).to_dict("records")
        field_names = (
            [proportion_value, "other"]
            if len(proportion_values) > 2
            else proportion_values
        )

    df = pd.DataFrame.from_records(class_counts)

    # filter less than 1% of total size categories
    category_sizes = (
        df[["_class_counts", category_field]].groupby(category_field).agg("sum")
    )
    df = df[
        df[category_field].isin(
            category_sizes[
                category_sizes["_class_counts"]
                / np.sum(category_sizes["_class_counts"].values)
                >= 0.01
            ].index
        )
    ]

    category_sizes = (
        df[["_class_counts", category_field]].groupby(category_field).agg("sum")
    )
    categories = sorted(list(set(df[proportion_field].values)))
    category_sizes_percentage = category_sizes["_class_counts"].values / np.sum(
        category_sizes["_class_counts"]
    )

    proportions = df.groupby(category_field).agg(lambda x: list(x))
    proportions = np.array(
        list(
            proportions.apply(
                lambda row: [
                    (
                        row["_class_counts"][row[proportion_field].index(cat)]
                        if cat in row[proportion_field]
                        else 0
                    )
                    for cat in categories
                ],
                axis=1,
            )
        )
    )
    proportions = proportions / proportions.sum(axis=1, keepdims=True)

    category_sizes_percentage = category_sizes_percentage * 0.95
    y_positions = np.cumsum(category_sizes_percentage)
    y_labels_positions = y_positions - category_sizes_percentage / 2

    fig = go.Figure()
    y_start = 0

    for i, (size, proportions) in enumerate(
        zip(category_sizes_percentage, proportions)
    ):
        x_start = 0
        for j, prop in enumerate(proportions):
            fig.add_shape(
                type="rect",
                x0=x_start,
                x1=x_start + prop,
                y0=y_start,
                y1=y_start + size,
                line=dict(color="black"),
                fillcolor=COLORS[j],
            )
            x_start += prop
        y_start += size

    x_start_label = 0

    for i, field_name in enumerate(field_names):
        fig.add_shape(
            type="rect",
            x0=x_start_label,
            x1=x_start_label + 0.1,
            y0=0.97,
            y1=1,
            line=dict(color="black"),
            fillcolor=COLORS[i],
            label=dict(text=f"{field_name}"),
        )
        x_start_label += 0.1

    title_text = None
    if name is not None and str(name).strip() != "":
        title_text = str(name)

    fig.update_layout(
        title=(
            dict(
                text=title_text,
                x=0.0,
                xanchor="left",
                y=0.985,
                yanchor="top",
                font=dict(color="white"),
            )
            if title_text
            else None
        ),
        margin=(dict(t=50) if title_text else None),
        autosize=True,
        height=max(len(category_sizes.index) * 30, 400),
        xaxis=dict(
            range=[0, 1],
            showgrid=False,
            title="Proportion by Sex",
            color="white",
            gridcolor="#222",
            zerolinecolor="#444",
            linecolor="#444",
        ),
        yaxis=dict(
            range=[0, 1],
            showgrid=False,
            tickmode="array",
            tickvals=y_labels_positions,
            ticktext=category_sizes.index,
            title="Categories",
            color="white",
            gridcolor="#222",
            zerolinecolor="#444",
            linecolor="#444",
        ),
        showlegend=False,
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font=dict(color="white"),
    )

    return fig
