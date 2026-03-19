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
        features, target = zip(*data)
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
