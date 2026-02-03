from collections import Counter
import math
import numpy as np
import pandas as pd
import torch

from ..metric import TabularMetric, MetricResult
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
            cluster="Representativeness",
            threshold=len(types),
        )
