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

        features = MultiLabelBinarizer().fit_transform(features)
        target = np.array(target)

        correlations = np.array(
            [
                pearsonr(features[:, j], target[:, i])[0]
                for j in range(features.shape[1])
                for i in range(target.shape[1])
            ]
        )
        return MetricResult(
            cluster="Informativeness",
            threshold=0,
            description="Pearson correlation between features and target",
            value=min(np.abs(correlations)),
        )
