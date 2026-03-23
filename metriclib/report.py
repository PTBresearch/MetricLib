from typing import List, Dict, Any, Optional, TypedDict
import hashlib
import json
from dateutil.parser import parse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from .util.util import add_bar, build_label_bar_figure, build_mosaique_label_figure

from .data import Dataset
from .metric import StreamMetric, TabularMetric, MetricResult
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
    def _mosaique_label_chart(
        dataset_dfs: List[pd.DataFrame],
        labels: List[pd.Series],
        filtered_dfs: List[pd.DataFrame] = None,
        chart_config: dict = {},
    ):
        figure = build_mosaique_label_figure(dataset_dfs, labels, **chart_config)
        return figure

    @staticmethod
    def _label_bar_chart(
        dataset_dfs: List[pd.DataFrame],
        labels: List[pd.Series],
        filtered_dfs: List[pd.DataFrame] = None,
        chart_config: dict = {},
    ):
        figure = build_label_bar_figure(dataset_dfs, labels, **chart_config)
        return figure

    @staticmethod
    def _continuous_bar_chart(
        dataset_dfs: List[pd.DataFrame],
        filtered_dfs: List[pd.DataFrame] = None,
        chart_config: dict = {},
    ):
        figure = go.Figure(data=[])
        field = chart_config.get("field")
        prepared = []
        all_keys = set()
        n_buckets = chart_config.get("n_buckets")

        for i, df in enumerate(dataset_dfs):
            if field not in df.columns:
                continue

            values = pd.to_numeric(df[field], errors="coerce").dropna()
            if filtered_dfs is not None and field in filtered_dfs[i].columns:
                filtered_values = pd.to_numeric(
                    filtered_dfs[i][field], errors="coerce"
                ).dropna()
            else:
                filtered_values = values

            if n_buckets is None:
                values = values.round(0).astype(int)
                filtered_values = filtered_values.round(0).astype(int)
            else:
                values = values.astype(float)
                filtered_values = filtered_values.astype(float)

            prepared.append((i, values, filtered_values))
            all_keys.update(values.tolist())
            all_keys.update(filtered_values.tolist())

        if not prepared or not all_keys:
            return figure

        if n_buckets is None:
            sorted_keys = sorted(all_keys)

            for i, values, filtered_values in prepared:
                hist_values = dict.fromkeys(sorted_keys, 0)
                hist_values_filtered = dict.fromkeys(sorted_keys, 0)
                hist_values.update(dict(values.value_counts().to_dict()))
                hist_values_filtered.update(
                    dict(filtered_values.value_counts().to_dict())
                )

                figure = add_bar(
                    f"dataset{i+1}",
                    sorted_keys,
                    figure,
                    hist_values,
                    hist_values_filtered,
                    yaxis_title="Counts",
                    xaxis_title=field,
                )

            return figure

        try:
            n_buckets = int(n_buckets)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_buckets must be an integer when provided.") from exc

        if n_buckets <= 0:
            raise ValueError("n_buckets must be > 0 when provided.")

        min_val = min(all_keys)
        max_val = max(all_keys)

        if min_val == max_val:
            bucket_edges = np.array([min_val, max_val + 1], dtype=np.float64)
        else:
            bucket_edges = np.linspace(
                min_val, max_val, n_buckets + 1, dtype=np.float64
            )

        bucket_count = len(bucket_edges) - 1

        def _format_edge(value: float) -> str:
            if float(value).is_integer():
                return str(int(value))
            return f"{value:.2f}".rstrip("0").rstrip(".")

        bucket_labels = [
            f"[{_format_edge(bucket_edges[idx])}, {_format_edge(bucket_edges[idx + 1])}{')' if idx < bucket_count - 1 else ']'}"
            for idx in range(bucket_count)
        ]

        def _bucket_counts(series: pd.Series) -> np.ndarray:
            if series.empty:
                return np.zeros(bucket_count, dtype=int)

            arr = series.to_numpy(dtype=np.float64)
            bucket_idx = np.digitize(arr, bucket_edges[1:-1], right=False)
            return np.bincount(bucket_idx, minlength=bucket_count)

        for i, values, filtered_values in prepared:
            counts = _bucket_counts(values)
            filtered_counts = _bucket_counts(filtered_values)
            hist_values = {
                bucket_labels[idx]: int(counts[idx]) for idx in range(bucket_count)
            }
            hist_values_filtered = {
                bucket_labels[idx]: int(filtered_counts[idx])
                for idx in range(bucket_count)
            }

            figure = add_bar(
                f"dataset{i+1}",
                bucket_labels,
                figure,
                hist_values,
                hist_values_filtered,
                yaxis_title="Counts",
                xaxis_title=field,
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
        field = chart_config.get("field")
        prepared = []
        all_keys = set()

        for i, df in enumerate(dataset_dfs):
            if field not in df.columns:
                continue

            values = df[field].dropna()
            if filtered_dfs is not None and field in filtered_dfs[i].columns:
                filtered_values = filtered_dfs[i][field].dropna()
            else:
                filtered_values = values

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
                values = values.dropna().astype(int)
                filtered_values = filtered_values.dropna().astype(int)
            else:
                parsed_years = values.map(_try_parse_year)
                parse_ratio = parsed_years.notna().mean() if len(parsed_years) else 0.0
                if parse_ratio >= 0.5:
                    values = parsed_years.dropna().astype(int)
                    filtered_values = (
                        filtered_values.map(_try_parse_year).dropna().astype(int)
                    )

            hist_values = values.value_counts().to_dict()
            hist_values_filtered = filtered_values.value_counts().to_dict()

            prepared.append((i, hist_values, hist_values_filtered))
            all_keys.update(hist_values.keys())
            all_keys.update(hist_values_filtered.keys())

        if not prepared or not all_keys:
            return figure

        try:
            numeric_keys = {k: float(k) for k in all_keys}
            sorted_keys = sorted(all_keys, key=lambda k: numeric_keys[k])
        except (TypeError, ValueError):
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
                xaxis_title=field,
            )

        return figure

    def __init__(self, datasets: List[Dataset]):
        self.datasets: List[Dataset] = datasets
        self.metrics: List[str, Dict[str, Any]] = []
        self.charts: List[Dict[str, Any]] = []
        self.scores: List[Dict[str, Optional[float]]] = []

        for _ in datasets:
            self.scores.append(
                {
                    "Measurement Process": 1.0,
                    "Timeliness": 1.0,
                    "Representativeness": 1.0,
                    "Informativeness": 1.0,
                    "Consistency": 1.0,
                }
            )

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

        if TabularMetric.registry.get(metric_name):
            self.metrics.append(
                {
                    "metric": TabularMetric.registry[metric_name],
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
        self.charts.append(
            {
                "name": name,
                "type": chart_type,
                "figure": "pending",
                "config": chart_config,
            }
        )

    def generate(self):
        class_name_to_index = {
            dataset.__class__.__name__: i for i, dataset in enumerate(self.datasets)
        }
        name_to_index = {
            dataset.name: i
            for i, dataset in enumerate(self.datasets)
            if getattr(dataset, "name", None) is not None
        }

        def _resolve_dataset_index(dataset_key: str) -> int:
            if dataset_key in class_name_to_index:
                return class_name_to_index[dataset_key]
            if dataset_key in name_to_index:
                return name_to_index[dataset_key]
            raise ValueError(
                f"Dataset {dataset_key} not found. Available: "
                f"{list(class_name_to_index.keys()) + list(name_to_index.keys())}"
            )

        def _update_score(dataset_index: int, result: MetricResult) -> None:
            if result.cluster and result.threshold is not None and result.threshold > 0:
                if result.cluster == "Representativeness":
                    print(result)
                metric_score = min(result.value / result.threshold, 1.0)
                if self.scores[dataset_index][result.cluster] > metric_score:
                    self.scores[dataset_index][result.cluster] = metric_score

        def _has_cached_stream_result(dataset: Dataset, metric_key: str) -> bool:
            md = getattr(dataset, "metadata", None)
            if md is None:
                return False
            if isinstance(md, pd.DataFrame):
                return metric_key in md.columns
            if isinstance(md, dict):
                return metric_key in md
            try:
                return metric_key in md
            except Exception:
                return False

        def _get_cached_stream_result(dataset: Dataset, metric_key: str):
            md = getattr(dataset, "metadata", None)
            if md is None:
                return None
            if isinstance(md, pd.DataFrame):
                return md[metric_key]
            if isinstance(md, dict):
                return md.get(metric_key)
            return md[metric_key]

        def _set_cached_stream_result(dataset: Dataset, metric_key: str, value) -> None:
            md = getattr(dataset, "metadata", None)
            if md is None:
                dataset.metadata = pd.DataFrame()
                md = dataset.metadata
            if isinstance(md, pd.DataFrame):
                md[metric_key] = value
                return
            if isinstance(md, dict):
                md[metric_key] = value
                return
            md[metric_key] = value

        def _build_stream_metric_key(metric_class, metric_config) -> str:
            try:
                config_payload = json.dumps(
                    metric_config if metric_config is not None else {},
                    sort_keys=True,
                    default=str,
                )
            except (TypeError, ValueError):
                config_payload = repr(metric_config)

            config_hash = hashlib.sha1(config_payload.encode("utf-8")).hexdigest()[:12]
            return f"{metric_class.__name__}__{config_hash}"

        stream_entries: List[Dict[str, Any]] = []
        stream_unique_by_dataset: Dict[int, Dict[str, Dict[str, Any]]] = {}

        for metric_idx, metric_info in enumerate(self.metrics):
            metric_class = metric_info["metric"]
            reference = metric_info["reference"]
            metric_config = metric_info["metric_config"]
            dataset_index = _resolve_dataset_index(metric_info["dataset"])
            dataset = self.datasets[dataset_index]

            if issubclass(metric_class, TabularMetric):
                result = metric_class().compute(
                    data=dataset.get_metadata(),
                    reference=reference,
                    metric_config=metric_config,
                )
                self.metrics[metric_idx]["result"] = result
                _update_score(dataset_index, result)
                continue

            if issubclass(metric_class, StreamMetric):
                metric_key = _build_stream_metric_key(metric_class, metric_config)
                stream_entries.append(
                    {
                        "metric_idx": metric_idx,
                        "dataset_index": dataset_index,
                        "metric_class": metric_class,
                        "metric_key": metric_key,
                        "reference": reference,
                        "metric_config": metric_config,
                    }
                )

                if dataset_index not in stream_unique_by_dataset:
                    stream_unique_by_dataset[dataset_index] = {}
                if metric_key not in stream_unique_by_dataset[dataset_index]:
                    stream_unique_by_dataset[dataset_index][metric_key] = {
                        "metric_class": metric_class,
                        "reference": reference,
                        "metric_config": metric_config,
                    }

        freshly_aggregated_results: Dict[int, Dict[str, Any]] = {}

        for dataset_index, metric_map in stream_unique_by_dataset.items():
            dataset = self.datasets[dataset_index]
            if getattr(dataset, "metadata", None) is None:
                dataset.metadata = pd.DataFrame()

            instance_by_key: Dict[str, StreamMetric] = {}
            keys_to_aggregate: List[str] = []

            for metric_key, info in metric_map.items():
                instance = info["metric_class"]()
                instance_by_key[metric_key] = instance
                if _has_cached_stream_result(dataset, metric_key):
                    instance.result = _get_cached_stream_result(dataset, metric_key)
                else:
                    keys_to_aggregate.append(metric_key)

            if keys_to_aggregate:
                for data_point in tqdm(dataset):
                    for metric_key in keys_to_aggregate:
                        info = metric_map[metric_key]
                        instance_by_key[metric_key].aggregate(
                            data_point,
                            info["reference"],
                            info["metric_config"],
                        )

                if dataset_index not in freshly_aggregated_results:
                    freshly_aggregated_results[dataset_index] = {}
                for metric_key in keys_to_aggregate:
                    freshly_aggregated_results[dataset_index][metric_key] = (
                        instance_by_key[metric_key].result
                    )
                    _set_cached_stream_result(
                        dataset, metric_key, instance_by_key[metric_key].result
                    )

        for entry in stream_entries:
            metric_idx = entry["metric_idx"]
            dataset_index = entry["dataset_index"]
            dataset = self.datasets[dataset_index]
            metric_key = entry["metric_key"]

            metric_instance = entry["metric_class"]()
            if (
                dataset_index in freshly_aggregated_results
                and metric_key in freshly_aggregated_results[dataset_index]
            ):
                metric_instance.result = freshly_aggregated_results[dataset_index][
                    metric_key
                ]
            elif _has_cached_stream_result(dataset, metric_key):
                metric_instance.result = _get_cached_stream_result(dataset, metric_key)

            result = metric_instance.compute(
                data=metric_instance.result,
                reference=entry["reference"],
                metric_config=entry["metric_config"],
            )
            self.metrics[metric_idx]["result"] = result
            _update_score(dataset_index, result)

        for name, chart_info in enumerate(self.charts):
            chart_type = chart_info["type"]
            chart_config = chart_info["config"]
            dataset_dfs = [ds.get_metadata() for ds in self.datasets]
            labels = [ds.get_labels() for ds in self.datasets]

            filtered_dfs = None
            if chart_config.get("filtered_metadata"):
                filters = chart_config["filtered_metadata"]
                if len(filters) != len(dataset_dfs):
                    raise ValueError(
                        "filtered_metadata must contain one entry per dataset."
                    )

                filtered_dfs = []
                for i, df in enumerate(dataset_dfs):
                    query = filters[i]
                    filtered_dfs.append(df.query(query) if query else df)

            if chart_type == "categorical_bar_chart":
                figure = self._categorical_bar_chart(
                    dataset_dfs=dataset_dfs,
                    filtered_dfs=filtered_dfs,
                    chart_config=chart_config,
                )
                self.charts[name]["figure"] = figure
            elif chart_type == "continuous_bar_chart":
                figure = self._continuous_bar_chart(
                    dataset_dfs=dataset_dfs,
                    filtered_dfs=filtered_dfs,
                    chart_config=chart_config,
                )
                self.charts[name]["figure"] = figure
            elif chart_type == "label_bar_chart":
                figure = self._label_bar_chart(
                    dataset_dfs=dataset_dfs,
                    labels=labels,
                    chart_config=chart_config,
                )
                self.charts[name]["figure"] = figure
            elif chart_type == "mosaique_label_chart":
                figure = self._mosaique_label_chart(
                    dataset_dfs=dataset_dfs,
                    labels=labels,
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
