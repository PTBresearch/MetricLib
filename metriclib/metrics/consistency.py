import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..metric import MetricResult, TabularMetric


class SyntacticConsistency(TabularMetric):
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


class MMD(TabularMetric):
    def compute(self, data: pd.DataFrame, reference: pd.DataFrame, metric_config=None):
        """Compute a simple MMD-like distance between grouped features.

        Configuration
        - groups: dict
            A dictionary with exactly one key where:
            * The key is the name of the grouping column in the DataFrame.
            * The value is a list of features column names whose
              aggregated statistics (e.g., mean) will be compared across the
              groups.
            Example:
                {"age group": ["20-30", "50-60"]}

        Parameters
        - data: pd.DataFrame
            Input dataset to compute the distance on.
        - reference: pd.DataFrame
            Reference dataset (not used in current implementation).
        - metric_config: dict | None
            Configuration with the keys described above.

        Returns
        - float
            A scalar norm of the difference between grouped means (placeholder
            implementation).
        """
        if not metric_config or type(metric_config.get("groups")) is not dict:
            raise ValueError(
                "Metric configuration must specify the 'field_groups' to compare."
            )

        if type(metric_config.get("feature_cols")) is not list:
            raise ValueError(
                "Metric configuration must specify the 'feature_cols' to select comparable features."
            )

        df_selected = data[
            metric_config.get("feature_cols")
            + [list(metric_config["groups"].keys())[0]]
        ].dropna()

        embedded_data = pd.DataFrame(df_selected.select_dtypes(include=["number"]))

        for col in df_selected.select_dtypes(exclude=["number"]).columns:
            if col == list(metric_config["groups"].keys())[0]:
                continue
            mlb = LabelEncoder()
            embedded_data[col] = pd.Series(mlb.fit_transform(data[col]))

        # normalize embedded data except group column
        for col in embedded_data.columns:
            if (
                col == list(metric_config["groups"].keys())[0]
                or embedded_data[col].mean() == 0
            ):
                continue
            embedded_data[col] = (
                embedded_data[col] - embedded_data[col].mean()
            ) / embedded_data[col].std()

        embedded_data = embedded_data.groupby(
            list(metric_config["groups"].keys())[0]
        ).mean()

        return MetricResult(
            description=f"MMD",
            value=np.linalg.norm(
                embedded_data.loc[list(metric_config["groups"].values())[0][0]]
                - embedded_data.loc[list(metric_config["groups"].values())[0][1]]
            ),
        )
