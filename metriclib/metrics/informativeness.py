import ast
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MultiLabelBinarizer

from ..metric import MetricResult, StreamMetric, TabularMetric


class Duplicates(TabularMetric):
    """Compute the proportion of duplicate rows in a tabular dataset."""

    def compute(self, data, reference=None, metric_config=None):
        """Return the proportion of duplicate rows in the dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            Input dataset for which duplicates will be counted.
        reference : pandas.DataFrame, optional
            Unused for this metric, included for API compatibility.
        metric_config : dict, optional
            Unused configuration dictionary, included for API compatibility.

        Returns
        -------
        MetricResult
            Result object with the proportion of duplicate rows.
        """

        duplicates = data.duplicated().sum()
        total_rows = data.shape[0]
        return MetricResult(
            cluster="Informativeness",
            threshold=0,
            description="Proportion of duplicate rows in the dataset",
            value=duplicates / total_rows,
        )


class PearsonCorrelation(StreamMetric):
    def aggregate(self, data_point, reference=None, metric_config=None):
        if not metric_config or "feature_columns" not in metric_config:
            raise ValueError("Metric configuration must include 'feature_columns' key.")

        features = [
            data_point[2][feature] for feature in metric_config["feature_columns"]
        ]

        return (features, data_point[1])

    def compute(self, data, reference=None, metric_config=None):
        if not metric_config or "feature_columns" not in metric_config:
            raise ValueError("Metric configuration must include 'feature_columns' key.")

        if data is None or len(data) == 0:
            raise ValueError("PearsonCorrelation requires non-empty data.")

        def _normalize_record(record):
            if isinstance(record, str):
                try:
                    record = ast.literal_eval(record)
                except (ValueError, SyntaxError):
                    return None

            # Cached JSON/dict-like payloads
            if isinstance(record, dict):
                if "features" in record and "target" in record:
                    return record["features"], record["target"]
                if "label" in record and "metadata" in record:
                    metadata = record.get("metadata")
                    if isinstance(metadata, dict):
                        features = [
                            metadata.get(feature)
                            for feature in metric_config["feature_columns"]
                        ]
                        return features, record["label"]

            # Native aggregate output: (features, target)
            if isinstance(record, (tuple, list)) and len(record) == 2:
                return record[0], record[1]

            # Backward-compatible fallback for cached/raw datapoints:
            # (x, target, metadata_dict)
            if isinstance(record, (tuple, list)) and len(record) >= 3:
                metadata = record[2]
                if isinstance(metadata, dict):
                    features = [
                        metadata.get(feature)
                        for feature in metric_config["feature_columns"]
                    ]
                    return features, record[1]

            return None

        normalized = []
        for record in data:
            parsed = _normalize_record(record)
            if parsed is not None:
                normalized.append(parsed)

        if not normalized:
            raise ValueError(
                "PearsonCorrelation expects records as (features, target) or raw datapoints (x, target, metadata)."
            )

        features, target = zip(*normalized)
        features = (
            MultiLabelBinarizer()
            .fit_transform([[str(value) for value in row] for row in features])
            .astype(np.float64)
        )

        try:
            target = np.asarray(target, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "PearsonCorrelation target must be numeric and rectangular."
            ) from exc

        if target.ndim == 1:
            target = target.reshape(-1, 1)
        elif target.ndim != 2:
            raise ValueError("PearsonCorrelation target must be a 1D or 2D array.")

        if features.shape[0] != target.shape[0]:
            raise ValueError(
                "PearsonCorrelation features and target must have the same number of samples."
            )

        if features.shape[0] < 2 or features.shape[1] == 0 or target.shape[1] == 0:
            value = 0.0
        else:
            correlations = np.asarray(
                [
                    pearsonr(features[:, j], target[:, i])[0]
                    for j in range(features.shape[1])
                    for i in range(target.shape[1])
                ],
                dtype=np.float64,
            )
            finite = np.abs(correlations[np.isfinite(correlations)])
            value = float(np.min(finite)) if finite.size else 0.0

        return MetricResult(
            cluster="Informativeness",
            threshold=0.00001,
            description="Pearson correlation between features and target",
            value=value,
        )
