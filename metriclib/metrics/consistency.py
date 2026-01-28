from ..metric import MetricResult, Metric
import pandas as pd


class SyntacticConsistency(Metric):
    def compute(self, data: pd.DataFrame, reference=None, metric_config=None):
        """Compute the number of syntactic violations in a column for a specified column from a reference dictionary.

        Parameters
        - data: pd.DataFrame
            Input DataFrame containing the target column.
        - reference: Optional[pd.DataFrame]
            Unused for this metric.
        - metric_config: dict
            Must include {"column": str, "allowed_values": List[Any]} specifying the target column and its allowed values.

        Returns
        - MetricResult
            A single result with description "{column} syntactic consistency" and scalar value
            representing the proportion of entries in the column that are within the allowed values.

        Raises
        - ValueError
            If the column is missing or if metric_config is improperly specified.
        """
        if (
            metric_config is None
            or "column" not in metric_config
            or "allowed_values" not in reference
        ):
            raise ValueError(
                "metric_config must include 'column' and 'allowed_values' keys"
            )
        col = metric_config["column"]
        allowed_values = set(reference["allowed_values"])
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        total_count = len(data)
        if total_count == 0:
            consistency = 1.0  # Define consistency as 1.0 for empty columns
        else:
            valid_count = data[col].isin(allowed_values).sum()
            consistency = valid_count / total_count

        return MetricResult(
            description=f"{col} syntactic consistency",
            value=consistency,
        )
