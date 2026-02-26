from datetime import datetime
import math
import pandas as pd

from ..metric import TabularMetric, MetricResult


class CurrencyHeinrich(TabularMetric):
    def compute(self, data, reference=None, metric_config=None):
        """Compute the Currency according to Heinrich.

        This metric models currency using an exponential decay of freshness
        over time since creation. It returns the mean currency across all rows.

        Required metric_config keys (strings of column names)
        - "created_at_field": name of the column with the record creation
          timestamp

        Optional metric_config keys
        - "A": float decay parameter (default: 1.0)

        Expectations for the referenced columns
        - Column referenced by "created_at_field": datetime-like values. Non-
          datetime values will be coerced; unparseable values become NaT and
          are ignored in the mean.

        Parameters
        - data: pd.DataFrame
            Input data containing the required fields.
        - reference: Optional[pd.DataFrame]
            Unused for this metric.
        - metric_config: dict | None
            Configuration with required/optional parameters and column names.

        Returns
        - MetricResult
            A single result containing the mean currency.
        """
        if metric_config is None or "created_at_field" not in metric_config:
            raise ValueError("metric_config must contain key 'created_at_field'")

        created_at = data[metric_config["created_at_field"]]

        if not pd.api.types.is_datetime64_any_dtype(created_at):
            created_at = pd.to_datetime(created_at, errors="coerce")

        A = metric_config.get("A", 1)

        created_at = created_at.dropna()
        mean_currency = created_at.apply(
            lambda x: math.exp(-A * (datetime.now().timestamp() - x.timestamp())),
        ).mean()

        return MetricResult(
            description="Currency by Heinrich",
            value=mean_currency,
            cluster="Timeliness",
            threshold=0.5,
        )
