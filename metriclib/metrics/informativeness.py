from ..metric import MetricResult, StreamMetric, TabularMetric


class Dublicates(TabularMetric):
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

        dublicates = data.duplicated().sum()
        total_rows = data.shape[0]
        return MetricResult(
            cluster="Informativeness",
            threshold=0,
            description="Proportion of duplicate rows in the dataset",
            value=dublicates / total_rows,
        )
