from typing import List, Dict, Any, Optional, TypedDict
from dateutil.parser import parse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from .util.util import add_bar

from .data import Dataset
from .metric import StreamMetric, TabularMetric
from .metrics import (
    timeliness,
    representativeness,
    consistency,
    informativeness,
    measurement_process,
)


class ReportType(TypedDict):
    scores: Dict[str, Optional[float]]
    metrics: Dict[str, Dict[str, Any]]
    charts: Dict[str, Any]


class Report:
    @staticmethod
    def _continuous_bar_chart(
        dataset_dfs: List[pd.DataFrame],
        filtered_dfs: List[pd.DataFrame] = None,
        chart_config: dict = {},
    ):
        figure = go.Figure(data=[])
        for i, df in enumerate(dataset_dfs):
            if not chart_config.get("field") in df.columns:
                continue

            if filtered_dfs:
                filtered_values = filtered_dfs[i][chart_config["field"]]
            else:
                filtered_values = df[chart_config["field"]]

            values = df[chart_config["field"]].round(0)
            values = values.dropna()
            values = values.astype(int)
            min_val = int(values.min())
            max_val = int(values.max())
            full_range = range(min_val, max_val + 1)
            hist_values = dict.fromkeys(full_range, 0)
            hist_values.update(dict(values.value_counts().to_dict()))
            sorted_keys = sorted(hist_values.keys())
            hist_values = {k: hist_values[k] for k in sorted_keys}

            figure = add_bar(
                f"dataset{i+1}",
                sorted_keys,
                figure,
                hist_values,
                hist_values,
                yaxis_title="Counts",
                xaxis_title=chart_config["field"],
            )

        return figure

    @staticmethod
    def _categorical_bar_chart(
        dataset_dfs: List[pd.DataFrame],
        filtered_dfs: List[pd.DataFrame] = None,
        chart_config: dict = {},
    ):
        """Build a bar chart for a categorical/ordinal field
        Parameters
        - chart_config: dict
                Configuration expected to contain:
                - "field": str
                        Column name in each dataset's metadata to visualize. The
                        column should be discrete (categorical/ordinal) for
                        meaningful histograms.
                - "filtered_metadata": list-like (optional)
                        A list with one entry per dataset (same order as
                        `self.datasets`). Each entry should be a pandas Series (or
                        array-like) representing a filtered subset of the values for
                        the same "field" to be shown alongside the unfiltered counts
        Behavior
        - For each dataset, computes value counts of `field` (dropping NaNs),
            sorts keys for stable x-axis, and (if provided) computes counts for
            the corresponding filtered values. Both are added to a shared
            Plotly Figure via `add_bar` as separate traces per dataset
        Notes
        - This method currently does not return or persist the constructed
            figure. If you need to reuse it later, consider storing it in
            `self.charts` outside this change.
        """
        figure = go.Figure(data=[])
        for i, df in enumerate(dataset_dfs):
            if not chart_config.get("field") in df.columns:
                continue

            if filtered_dfs:
                filtered_values = filtered_dfs[i][chart_config["field"]]
            else:
                filtered_values = df[chart_config["field"]]

            values = df[chart_config["field"]]
            values = values.dropna()

            # Robust date handling: use dateutil.parse for string dates
            is_dt_dtype = pd.api.types.is_datetime64_any_dtype(
                values
            ) or pd.api.types.is_datetime64tz_dtype(values)

            def _try_parse_year(v):
                try:
                    return parse(v, fuzzy=False).year if isinstance(v, str) else np.nan
                except Exception:
                    return np.nan

            if is_dt_dtype:
                values = pd.to_datetime(values, errors="coerce").dt.year
                filtered_values = pd.to_datetime(
                    filtered_values, errors="coerce"
                ).dt.year
            else:
                parsed_years = values.map(_try_parse_year)
                parse_ratio = parsed_years.notna().mean() if len(parsed_years) else 0.0
                if parse_ratio >= 0.5:
                    values = parsed_years.dropna().astype(int)
                    filtered_values = (
                        filtered_values.map(_try_parse_year).dropna().astype(int)
                    )

            hist_values = dict(sorted(values.value_counts().to_dict().items()))
            hist_values_filtered = dict(
                sorted(filtered_values.value_counts().to_dict().items())
            )

            sorted_keys = sorted(hist_values.keys())

            figure = add_bar(
                f"dataset{i+1}",
                sorted_keys,
                figure,
                hist_values,
                hist_values_filtered,
                yaxis_title="Counts",
                xaxis_title=chart_config["field"],
            )

        return figure

    def __init__(self, datasets: List[Dataset]):
        self.datasets: List[Dataset] = datasets
        self.metrics: List[str, Dict[str, Any]] = []
        self.charts: List[Dict[str, Any]] = []
        self.scores: List[Dict[str, Optional[float]]] = [
            {
                "Measurement Process": 0.0,
                "Timeliness": 0.0,
                "Representativeness": 0.0,
                "Informativeness": 0.0,
                "Consistency": 0.0,
            }
            for _ in datasets
        ]

    def get_available_metrics(self) -> List[str]:
        return list(TabularMetric.registry.keys()) + list(StreamMetric.registry.keys())

    def add_metric(
        self,
        metric_name: str,
        reference: pd.DataFrame = None,
        metric_config: dict = None,
        dataset_name=None,
        name: str = None,
    ):
        if len(self.datasets) > 1 and dataset_name is None:
            raise ValueError(
                "Multiple datasets present. Please specify the dataset_name for the metric."
            )

        if Metric.registry.get(metric_name):
            self.metrics.append(
                {
                    "metric": Metric.registry[metric_name],
                    "reference": reference,
                    "metric_config": metric_config,
                    "dataset": (
                        dataset_name
                        if dataset_name
                        else self.datasets[0].__class__.__name__
                    ),
                    "name": name,
                }
            )
        elif StreamMetric.registry.get(metric_name):
            self.metrics.append(
                {
                    "metric": StreamMetric.registry[metric_name],
                    "reference": reference,
                    "metric_config": metric_config,
                    "dataset": (
                        dataset_name
                        if dataset_name
                        else self.datasets[0].__class__.__name__
                    ),
                    "name": name,
                }
            )
        else:
            raise ValueError(f"Metric {metric_name} not found in registry.")

    def add_chart(self, name, chart_type: str, chart_config: dict):
        dataset_dfs = [ds.get_metadata() for ds in self.datasets]
        filtered_dfs = []
        if chart_config.get("filtered_metadata"):
            for i, ds in enumerate(dataset_dfs):
                query = chart_config["filtered_metadata"][i]
                if query:
                    filtered_dfs.append(ds.get_metadata().query(query))
                else:
                    filtered_dfs.append(ds.get_metadata())
        self.charts.append(
            {
                "name": name,
                "type": chart_type,
                "figure": "pending",
                "config": chart_config,
            }
        )

    def generate(self):
        for name, metric_info in enumerate(self.metrics):
            metric_class = metric_info["metric"]
            reference = metric_info["reference"]
            metric_config = metric_info["metric_config"]

            if metric_info["dataset"] in [d.__class__.__name__ for d in self.datasets]:
                index = [d.__class__.__name__ for d in self.datasets].index(
                    metric_info["dataset"]
                )
            else:
                index = [d.name for d in self.datasets].index(metric_info["dataset"])

            dataset = self.datasets[index]

            metric_instance = metric_class()
            if issubclass(metric_class, Metric):
                data_df = dataset.get_metadata()
                result = metric_instance.compute(
                    data=data_df,
                    reference=reference,
                    metric_config=metric_config,
                )
                self.metrics[name]["result"] = result
            elif issubclass(metric_class, StreamMetric):
                for data_point in tqdm(dataset):
                    metric_instance.aggregate(
                        data_point,
                        reference,
                        metric_config,
                    )

                result = metric_instance.compute(
                    data=metric_instance.result,
                    reference=reference,
                    metric_config=metric_config,
                )
                self.metrics[name]["result"] = result

            if result.cluster and result.threshold:
                deviation = (
                    1 + min(result.value - result.threshold, 0) / result.threshold
                )
                if self.scores[index][result.cluster] < deviation:
                    self.scores[index][result.cluster] = deviation

        for name, chart_info in enumerate(self.charts):
            chart_type = chart_info["type"]
            chart_config = chart_info["config"]
            dataset_dfs = [ds.get_metadata() for ds in self.datasets]
            if chart_type == "categorical_bar_chart":
                figure = self._categorical_bar_chart(
                    dataset_dfs=dataset_dfs,
                    chart_config=chart_config,
                )
                self.charts[name]["figure"] = figure
            elif chart_type == "continuous_bar_chart":
                figure = self._continuous_bar_chart(
                    dataset_dfs=dataset_dfs,
                    chart_config=chart_config,
                )
                self.charts[name]["figure"] = figure
            else:
                raise ValueError(f"Chart type {chart_type} not recognized.")

        return self.metrics, self.charts, self.scores

    def to_dict(self) -> ReportType:
        return {
            "scores": self.scores,
            "metrics": self.metrics,
            "charts": self.charts,
        }
