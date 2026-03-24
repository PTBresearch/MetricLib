import pandas as pd
import plotly.graph_objects as go
import numpy as np
import torch
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Union, Any

COLORS = [
    "#FF6347",
    "#008080",
    "#1E90FF",
    "#FFD700",
]

HEATMAP_COLORSCALE = [
    [0.0, "#008080"],
    [0.25, "#2A4A4A"],
    [0.5, "#111111"],
    [0.75, "#4A2A2A"],
    [1.0, "#FF6347"],
]


def _tensor_labels_to_class_series(label_values) -> pd.Series:
    if not isinstance(label_values, list):
        raise TypeError("Label figures expect a list of torch.Tensor labels.")

    class_lists = []
    for row in label_values:
        if not isinstance(row, torch.Tensor):
            raise TypeError("Label figures expect a list of torch.Tensor labels.")

        flattened = row.detach().cpu().reshape(-1)

        if flattened.numel() == 0:
            class_lists.append([])
            continue

        if torch.is_floating_point(flattened):
            flattened = flattened[~torch.isnan(flattened)]

        if flattened.numel() == 0:
            class_lists.append([])
            continue

        unique_values = torch.unique(flattened)
        if all(value.item() in {0, 1} for value in unique_values):
            class_lists.append(
                [f"Class {index}" for index in torch.where(flattened == 1)[0].tolist()]
            )
            continue

        class_lists.append(
            [f"Class {value}" for value in flattened.to(torch.int64).tolist()]
        )

    return pd.Series(class_lists)


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


def build_label_bar_figure(dataset_dfs, labels) -> go.Figure:
    """Build a label bar chart equivalent to categorical_bar_chart.

    Labels are expected to be lists of torch tensors and are exploded to class
    names (e.g. "Class 0", "Class 1") before counting.
    """
    figure = go.Figure(data=[])
    prepared = []
    all_keys = set()

    for i, _ in enumerate(dataset_dfs):
        if i >= len(labels):
            continue

        label_series = _tensor_labels_to_class_series(labels[i])

        values = label_series.explode().dropna()
        filtered_values = values

        hist_values = values.value_counts().to_dict()
        hist_values_filtered = filtered_values.value_counts().to_dict()

        prepared.append((i, hist_values, hist_values_filtered))
        all_keys.update(hist_values.keys())
        all_keys.update(hist_values_filtered.keys())

    if not prepared or not all_keys:
        return figure

    try:
        numeric_keys = {k: float(str(k).split(" ")[-1]) for k in all_keys}
        sorted_keys = sorted(all_keys, key=lambda k: numeric_keys[k])
    except (TypeError, ValueError, IndexError):
        sorted_keys = sorted(all_keys, key=lambda k: str(k).casefold())

    for i, hist_values, hist_values_filtered in prepared:
        aligned_hist = {k: hist_values.get(k, 0) for k in sorted_keys}
        aligned_filtered = {k: hist_values_filtered.get(k, 0) for k in sorted_keys}

        figure = add_bar(
            f"dataset{i+1}",
            sorted_keys,
            figure,
            aligned_hist,
            aligned_filtered,
            yaxis_title="Counts",
            xaxis_title="labels",
        )

    return figure


def build_label_heatmap_figure(
    dataset_dfs: List[pd.DataFrame],
    labels: List[list],
    fields: List[str],
) -> go.Figure:
    """Build a single heatmap figure showing field-label correlation for all datasets.

    Each dataset gets its own subplot (stacked vertically). Rows are the metadata
    fields, columns are label classes.

    Returns a single Plotly Figure.
    """

    def _is_scalar_column(series: pd.Series) -> bool:
        sample = series.dropna().head(50)
        return not sample.apply(
            lambda v: isinstance(v, (list, tuple, dict, torch.Tensor, np.ndarray))
        ).any()

    # Pre-compute per-dataset data
    dataset_results = []
    for i, df in enumerate(dataset_dfs):
        if i >= len(labels):
            dataset_results.append(None)
            continue

        raw_labels = labels[i]
        if isinstance(raw_labels, torch.Tensor):
            raw_labels = list(raw_labels)
        elif isinstance(raw_labels, pd.Series):
            raw_labels = raw_labels.tolist()

        label_series = _tensor_labels_to_class_series(raw_labels).reset_index(drop=True)
        meta = df.reset_index(drop=True)

        if len(label_series) != len(meta):
            min_len = min(len(label_series), len(meta))
            label_series = label_series.iloc[:min_len]
            meta = meta.iloc[:min_len].copy()

        label_dummies = label_series.explode().str.strip().to_frame("class")
        label_dummies["_idx"] = label_dummies.index
        label_dummies = label_dummies.dropna(subset=["class"])
        label_dummies = label_dummies[label_dummies["class"] != ""]
        class_names = sorted(label_dummies["class"].unique(), key=lambda k: str(k))

        if not class_names:
            dataset_results.append(None)
            continue

        indicator_df = pd.DataFrame(0, index=meta.index, columns=class_names)
        for cls in class_names:
            rows_with_cls = label_dummies.loc[
                label_dummies["class"] == cls, "_idx"
            ].unique()
            indicator_df.loc[indicator_df.index.isin(rows_with_cls), cls] = 1

        valid_fields = [f for f in fields if f in meta.columns] if fields else []
        if not valid_fields:
            for c in meta.columns:
                try:
                    if not _is_scalar_column(meta[c]):
                        continue
                    if pd.to_numeric(meta[c], errors="coerce").notna().mean() >= 0.5:
                        valid_fields.append(c)
                except (TypeError, ValueError):
                    continue

        if not valid_fields:
            dataset_results.append(None)
            continue

        field_values = pd.DataFrame(index=meta.index)
        usable_fields = []
        for f in valid_fields:
            col = meta[f]
            try:
                if not _is_scalar_column(col):
                    continue
                numeric_col = pd.to_numeric(col, errors="coerce")
                if numeric_col.notna().sum() / max(len(numeric_col), 1) >= 0.5:
                    field_values[f] = numeric_col
                else:
                    field_values[f] = col.astype("category").cat.codes.replace(
                        -1, np.nan
                    )
                usable_fields.append(f)
            except (TypeError, ValueError):
                continue
        valid_fields = usable_fields

        if not valid_fields:
            dataset_results.append(None)
            continue

        corr_matrix = np.full((len(valid_fields), len(class_names)), np.nan)
        for fi, f in enumerate(valid_fields):
            fv = field_values[f]
            mask = fv.notna()
            if mask.sum() < 3:
                continue
            for ci, cls in enumerate(class_names):
                iv = indicator_df[cls].loc[mask].astype(float)
                fv_clean = fv.loc[mask].astype(float)
                if iv.std() == 0 or fv_clean.std() == 0:
                    corr_matrix[fi, ci] = 0.0
                else:
                    corr_matrix[fi, ci] = np.corrcoef(fv_clean, iv)[0, 1]

        text_matrix = np.where(
            np.isnan(corr_matrix), "", np.char.mod("%.2f", corr_matrix)
        )

        dataset_results.append(
            {
                "class_names": class_names,
                "valid_fields": valid_fields,
                "corr_matrix": corr_matrix,
                "text_matrix": text_matrix,
            }
        )

    valid_results = [(i, r) for i, r in enumerate(dataset_results) if r is not None]

    if not valid_results:
        return go.Figure()

    n_subplots = len(valid_results)
    subplot_titles = [f"Dataset {i + 1}" for i, _ in valid_results]
    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15 / max(n_subplots, 1),
    )

    for row_idx, (ds_idx, result) in enumerate(valid_results, start=1):
        show_colorbar = row_idx == 1
        fig.add_trace(
            go.Heatmap(
                z=result["corr_matrix"],
                x=[str(c) for c in result["class_names"]],
                y=result["valid_fields"],
                text=result["text_matrix"],
                texttemplate="%{text}",
                textfont=dict(color="white"),
                colorscale=HEATMAP_COLORSCALE,
                zmid=0,
                zmin=-1,
                zmax=1,
                showscale=show_colorbar,
                colorbar=(
                    dict(
                        title=dict(text="Correlation", font=dict(color="white")),
                        tickfont=dict(color="white"),
                    )
                    if show_colorbar
                    else None
                ),
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(
            autorange="reversed", color="white", linecolor="#333", row=row_idx, col=1
        )
        fig.update_xaxes(color="white", linecolor="#333", row=row_idx, col=1)

    # Label bottom x-axis only
    fig.update_xaxes(title_text="Label Class", row=n_subplots, col=1)

    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=80, r=40, t=60, b=60),
        height=max(300 * n_subplots, 400),
    )

    # Style subplot titles white
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(color="white")

    return fig


def build_mosaique_label_figure(
    filtered_metadata, labels, index, category_field, proportion_field, name
) -> go.Figure:
    metadata_df = filtered_metadata[index].copy()

    label_series = _tensor_labels_to_class_series(labels[index]).reset_index(drop=True)
    metadata_df = metadata_df.reset_index(drop=True)

    if len(label_series) != len(metadata_df):
        min_len = min(len(label_series), len(metadata_df))
        label_series = label_series.iloc[:min_len]
        metadata_df = metadata_df.iloc[:min_len].copy()

    metadata_df["labels"] = label_series

    proportion_values = metadata_df[proportion_field].unique()
    for proportion_value in proportion_values:
        metadata = metadata_df
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
