from collections import Counter
import ast
import math
import re
import numpy as np
import pandas as pd
import torch
import ot
from scipy.spatial.distance import cdist

from ..metric import TabularMetric, StreamMetric, MetricResult
from ..util.util import add_bar


class Range(TabularMetric):
    def compute(self, data: pd.DataFrame, reference=None, metric_config=None):
        """Compute range (max - min) for a single numeric column.

        Parameters
        - data: pd.DataFrame
            Input DataFrame containing numeric columns.
        - reference: Optional[pd.DataFrame]
            Unused for this metric.
        - metric_config: dict
            Must include ``{"column": str}`` to compute the range for that column.

        Returns
                - MetricResult
                        A single result with description ``"{column} range"`` and a scalar value.

        Raises
        - ValueError
            If any of the specified columns are not numeric.
        """
        if metric_config is None or "column" not in metric_config:
            raise ValueError("metric_config must include a 'column' key")
        col = metric_config["column"]
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError("Specified column must be numeric to compute Range.")
        return MetricResult(
            description=f"{col} range",
            value=data[col].max() - data[col].min(),
        )


class IQR(TabularMetric):
    def compute(self, data: pd.DataFrame, reference=None, metric_config=None):
        """Compute interquartile range (Q3 - Q1) for a single numeric column.

        Parameters
        - data: pd.DataFrame
            Input DataFrame containing numeric columns.
        - reference: Optional[pd.DataFrame]
            Unused for this metric.
        - metric_config: dict
            Must include {"column": str} specifying the target column.

        Returns
        - MetricResult
            A single result with description "{column} IQR" and scalar value
            data[column].quantile(0.75) - data[column].quantile(0.25).

        Raises
        - ValueError
            If the column is missing or not numeric.
        """
        if metric_config is None or "column" not in metric_config:
            raise ValueError("metric_config must include a 'column' key")
        col = metric_config["column"]
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError("Specified column must be numeric to compute IQR.")
        value = data[col].quantile(0.75) - data[col].quantile(0.25)
        return MetricResult(description=f"{col} IQR", value=value)


class StdDev(TabularMetric):
    def compute(self, data: pd.DataFrame, reference=None, metric_config=None):
        """Compute sample standard deviation (ddof=1) for a single numeric column.

        Parameters
        - data: pd.DataFrame
            Input DataFrame containing numeric columns.
        - reference: Optional[pd.DataFrame]
            Unused for this metric.
        - metric_config: dict
            Must include {"column": str} specifying the target column.

        Returns
        - MetricResult
            A single result with description "{column} StdDev" and scalar value
            data[column].std().

        Raises
        - ValueError
            If the column is missing or not numeric.
        """
        if metric_config is None or "column" not in metric_config:
            raise ValueError("metric_config must include a 'column' key")
        col = metric_config["column"]
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError("Specified column must be numeric to compute StdDev.")
        value = (
            data[col].std(),
            data[col].mean(),
        )
        return MetricResult(description=f"{col} StdDev", value=value)


class HillNumbers(TabularMetric):
    def compute(self, data: pd.DataFrame, reference=None, metric_config=None):
        """Compute Hill number (order q) for a single categorical column.

        Parameters
        - data: pd.DataFrame
            Input DataFrame containing categorical columns.
        - reference: Optional[pd.DataFrame]
            Unused.
        - metric_config: dict
            Must include {"column": str, "types": list} where types lists the expected categories.
            Optional: "q" (float, default 2), "normalize" (bool) to scale relative to category count.

        Returns
        - MetricResult
            A single result with description "Hill Numbers q={q} for {column}" and scalar value.

        Raises
        - ValueError
            If required keys are missing or column not found.
        """
        if (
            metric_config is None
            or "column" not in metric_config
            or "types" not in metric_config
            or not isinstance(metric_config.get("types"), (list, tuple))
        ):
            raise ValueError(
                "metric_config must include 'column' and 'types' (list of categories)"
            )
        col = metric_config["column"]
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        types = list(metric_config["types"])
        q = metric_config.get("q", 2)

        counts = (
            data[col]
            .value_counts(dropna=False, normalize=True)
            .reindex(types, fill_value=0)
        )

        def _hill_number(c, q):
            return (c**q).sum() ** (1 / (1 - q))

        value = _hill_number(counts, q)

        if metric_config.get("normalize"):
            value = value / (
                len(types) ** (1 / (1 - q)) * (1 / len(types) ** (q / (1 - q)))
            )

        return MetricResult(
            description=f"Hill Numbers q={q} for {col}",
            value=value,
            threshold=len(types),
        )


class WassersteinDistance1DTabular(TabularMetric):
    def compute(self, data: pd.DataFrame, reference: pd.Series, metric_config=None):
        """
        Compute 1D Wasserstein distance between a categorical column and a
        reference relative histogram.

        metric_config must include:
            {"column": "col1"}

        `reference` is expected to be a relative histogram as a pd.Series
        with a simple Index of category labels, e.g.:
            reference = raw_df[col].value_counts(normalize=True)
        """

        if metric_config is None or "column" not in metric_config:
            raise ValueError("metric_config must include a 'column' key")

        col = metric_config["column"]

        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data DataFrame.")
        if pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column '{col}' must be categorical (non-numeric).")

        data_ = data[col].dropna()

        if len(data_) == 0:
            raise ValueError("Data is empty after dropping NaNs.")

        # Validate reference histogram
        if not isinstance(reference, pd.Series):
            raise ValueError(
                f"reference must be a pd.Series relative histogram, got {type(reference)}."
            )
        if isinstance(reference.index, pd.MultiIndex):
            raise ValueError(
                "reference must have a simple Index of category labels, not a MultiIndex."
            )
        if reference.values.min() < 0:
            raise ValueError("reference histogram contains negative values.")
        if not np.isclose(reference.sum(), 1.0, atol=1e-3):
            raise ValueError(
                f"reference histogram must sum to 1.0 (got {reference.sum():.4f}). "
                "Normalize it first: ref / ref.sum()"
            )

        joint_data = data_.value_counts()

        joint_data, joint_ref = joint_data.align(reference, fill_value=0)

        p = joint_data.values.astype(float)
        q = joint_ref.values.astype(float)

        p /= p.sum()
        q /= q.sum()

        n = len(p)

        M = np.ones((n, n)) - np.eye(n)

        transport_matrix = ot.emd(p, q, M)
        value = float((transport_matrix * M).sum())

        return MetricResult(
            description=f"1D Wasserstein Distance for {col} (categorical)",
            value=value,
            cluster="Representativeness",
            threshold=0,
        )


class WassersteinDistance2DTabular(TabularMetric):
    def compute(self, data: pd.DataFrame, reference: pd.Series, metric_config=None):
        """
        Compute 2D Wasserstein distance between two categorical columns.

        metric_config must include:
            {"columns": ["col1", "col2"]}

        `reference` is expected to be a relative joint histogram as a pd.Series
        with a MultiIndex of (col1, col2) pairs, e.g.:
            reference = raw_df.groupby([col1, col2]).size()
            reference = reference / reference.sum()
        """

        if metric_config is None or "columns" not in metric_config:
            raise ValueError("metric_config must include a 'columns' key")

        cols = metric_config["columns"]

        if len(cols) != 2:
            raise ValueError("'columns' must contain exactly two column names")

        col1, col2 = cols

        for col in cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data DataFrame.")
            if pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column '{col}' must be categorical (non-numeric).")

        data_ = data[cols].dropna()

        if len(data_) == 0:
            raise ValueError("Data is empty after dropping NaNs.")

        if not isinstance(reference, pd.Series):
            raise ValueError(
                f"reference must be a pd.Series relative histogram, got {type(reference)}."
            )
        if not isinstance(reference.index, pd.MultiIndex):
            raise ValueError("reference must have a MultiIndex of (col1, col2) pairs.")
        if reference.values.min() < 0:
            raise ValueError("reference histogram contains negative values.")
        if not np.isclose(reference.sum(), 1.0, atol=1e-3):
            raise ValueError(
                f"reference histogram must sum to 1.0 (got {reference.sum():.4f}). "
                "Normalize it first: ref / ref.sum()"
            )

        joint_data = data_.groupby([col1, col2]).size()

        joint_data, joint_ref = joint_data.align(reference, fill_value=0)

        p = joint_data.values.astype(float)
        q = joint_ref.values.astype(float)

        p /= p.sum()
        q /= q.sum()

        n = len(p)

        M = np.ones((n, n)) - np.eye(n)

        transport_matrix = ot.emd(p, q, M)
        value = float((transport_matrix * M).sum())

        return MetricResult(
            description=f"2D Wasserstein Distance for {col1} & {col2} (categorical)",
            value=value,
            cluster="Representativeness",
            threshold=0,
        )


class MultiClassGeneralizedImbalanceRatio(StreamMetric):
    def aggregate(self, data_point, reference=None, metric_config=None):
        return data_point[1]

    def compute(self, data, reference, metric_config):
        def _parse_single_class(value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return []

            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.ndim == 0:
                    value = value.item()
                else:
                    value = value.tolist()

            if isinstance(value, str):
                stripped = value.strip()
                try:
                    parsed = ast.literal_eval(stripped)
                    if parsed is value:
                        parsed = stripped
                    value = parsed
                except (ValueError, SyntaxError):
                    match = re.search(r"-?\d+", stripped)
                    if match:
                        return [int(match.group())]
                    return []

            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, pd.Series):
                value = value.tolist()

            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return []

                if all(
                    isinstance(v, (int, float, np.integer, np.floating)) for v in value
                ):
                    numeric = np.asarray(value, dtype=float)
                    if numeric.ndim == 1 and set(np.unique(numeric)).issubset(
                        {0.0, 1.0}
                    ):
                        return np.where(numeric == 1.0)[0].astype(int).tolist()
                    return [int(v) for v in numeric.tolist() if not np.isnan(v)]

                parsed_classes = []
                for item in value:
                    parsed_classes.extend(_parse_single_class(item))
                return parsed_classes

            if isinstance(value, (int, np.integer)):
                return [int(value)]

            if isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    return []
                return [int(value)]

            return []

        if isinstance(data, pd.Series):
            records = data.tolist()
        else:
            records = list(data)

        classes = []
        for record in records:
            classes.extend(_parse_single_class(record))

        if len(classes) == 0:
            raise ValueError(
                "MultiClassGeneralizedImbalanceRatio requires parseable class labels."
            )

        counts = pd.Series(classes).value_counts()
        min_count = counts.min()
        if min_count == 0:
            raise ValueError(
                "MultiClassGeneralizedImbalanceRatio is undefined when min class count is 0."
            )
        value = 1 / (counts.max() / min_count)
        return MetricResult(
            description="MultiClass Generalized Imbalance Ratio",
            value=float(value),
            threshold=1.0,
        )


class MultiLabelGeneralizedImbalanceRatio(StreamMetric):
    def aggregate(self, data_point, reference=None, metric_config=None):
        return data_point[1]

    def compute(self, data, reference, metric_config):
        counts = np.unique(np.array(data), return_counts=True)[1]
        min_count = min(counts)
        if min_count == 0:
            raise ValueError(
                "MultiLabelGeneralizedImbalanceRatio is undefined when min label count is 0."
            )
        value = 1 / (max(counts) / min_count)
        return MetricResult(
            description="MultiLabel Generalized Imbalance Ratio",
            value=value,
            cluster="Representativeness",
            threshold=0.5,
        )


class MultiClassDemographicParity(StreamMetric):
    def aggregate(self, data_point, reference=None, metric_config=None):
        labels = torch.where(torch.tensor(data_point[1]) == 1)[0].tolist()
        return (
            labels,
            data_point[2][metric_config["protected_attribute"]],
        )

    def compute(self, data, reference, metric_config):
        def _parse_record(record):
            if isinstance(record, str):
                try:
                    record = ast.literal_eval(record)
                except (ValueError, SyntaxError):
                    return None

            if isinstance(record, np.ndarray):
                record = record.tolist()
            elif isinstance(record, pd.Series):
                record = record.tolist()

            if isinstance(record, dict):
                if "label" in record and "protected_attribute" in record:
                    return record["label"], record["protected_attribute"]
                return None

            if isinstance(record, (tuple, list)) and len(record) == 2:
                return record[0], record[1]

            return None

        def _normalize_labels(labels):
            if labels is None:
                return []

            if isinstance(labels, str):
                try:
                    parsed = ast.literal_eval(labels)
                    if isinstance(parsed, (list, tuple, np.ndarray, pd.Series)):
                        labels = parsed
                    else:
                        labels = [parsed]
                except (ValueError, SyntaxError):
                    labels = [labels]

            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            elif isinstance(labels, pd.Series):
                labels = labels.tolist()

            if isinstance(labels, (list, tuple)):
                return list(labels)

            if pd.isna(labels):
                return []

            return [labels]

        if isinstance(data, pd.Series):
            records = data.tolist()
        else:
            records = list(data)

        normalized_records = []
        for record in records:
            parsed = _parse_record(record)
            if parsed is None:
                continue

            labels, protected_attribute = parsed
            normalized_records.append((_normalize_labels(labels), protected_attribute))

        if not normalized_records:
            raise ValueError(
                "MultiClassDemographicParity expects items as (labels, protected_attribute)."
            )

        df = pd.DataFrame(normalized_records, columns=["label", "protected_attribute"])
        df = df.explode("label")
        df = df.dropna(subset=["label", "protected_attribute"])

        if df.empty:
            raise ValueError(
                "MultiClassDemographicParity is undefined when no positive labels are present."
            )

        group_counts = (
            df.groupby(["protected_attribute", "label"]).size().unstack(fill_value=0)
        )

        group_mins = group_counts.min(axis=0)
        group_maxs = group_counts.max(axis=0)

        value = 1.0 / group_maxs.div(group_mins).max()

        if value == float("inf"):
            raise ValueError(
                "DemographicParity is undefined when any group has 0 positive labels."
            )

        return MetricResult(
            description="Mean Multiclass Demographic Parity Ratio",
            value=value,
            cluster="Representativeness",
            threshold=0.3,
        )


class MultiLabelDemographicParity(StreamMetric):
    """Demographic parity ratio for multi-label data.

    Computes, per label, the ratio of the maximum to minimum count across
    protected groups, then returns the maximum of those ratios.
    """

    def aggregate(self, data_point, reference=None, metric_config=None):
        """Collect (label, protected_attribute) pairs from a single data point."""
        return (
            data_point[1],
            data_point[2][metric_config["protected_attribute"]],
        )

    def compute(self, data, reference, metric_config):
        """Compute the demographic parity ratio.

        Raises ValueError if any protected group has zero samples for a label.
        """
        if metric_config is None or "protected_attribute" not in metric_config:
            raise ValueError("metric_config must include 'protected_attribute'.")

        if data is None or len(data) == 0:
            raise ValueError("MultiLabelDemographicParity requires non-empty data.")

        df = pd.DataFrame(data, columns=["label", "protected_attribute"])

        group_counts = (
            df.groupby(["protected_attribute", "label"]).size().unstack(fill_value=0)
        )

        group_mins = group_counts.min(axis=0)
        group_maxs = group_counts.max(axis=0)

        if (group_mins == 0).any():
            raise ValueError(
                "MultiLabelDemographicParity is undefined when any group has 0 samples for a label."
            )

        value = 1.0 / group_maxs.div(group_mins).max()

        return MetricResult(
            description="Mean Multilabel Demographic Parity Ratio",
            value=value,
            threshold=0.5,
        )


class Resolution(StreamMetric):
    def aggregate(self, datapoint, reference=None, metric_config=None):
        return datapoint[0].shape

    def compute(self, data, reference, metric_config):
        def _to_hashable(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                return tuple(_to_hashable(item) for item in value)
            if isinstance(value, np.generic):
                return value.item()
            return value

        distinct_resolutions = len(Counter(_to_hashable(value) for value in data))

        return MetricResult(
            cluster=None,
            description="Number of distinct resolutions in the dataset",
            value=distinct_resolutions,
        )


class DatasetSize(TabularMetric):
    def compute(self, data, reference=None, metric_config=None):
        return MetricResult(
            cluster=None,
            description="Dataset size (number of samples)",
            value=len(data),
        )
