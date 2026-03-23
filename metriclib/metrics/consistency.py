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

        group_items = list(metric_config["groups"].items())
        if len(group_items) != 1:
            raise ValueError("'groups' must contain exactly one grouping column.")

        group_col, group_values = group_items[0]
        if not isinstance(group_values, list) or len(group_values) != 2:
            raise ValueError(
                "'groups' must map the grouping column to exactly two group values."
            )

        required_cols = metric_config.get("feature_cols", []) + [group_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Columns not found in DataFrame: {', '.join(map(str, missing_cols))}"
            )

        df_selected = data[required_cols].dropna()

        if df_selected.empty:
            return MetricResult(
                description=f"MMD undefined: no non-null rows for comparison on {group_col}",
                value=None,
                cluster="Consistency",
                threshold=None,
            )

        embedded_data = pd.DataFrame(index=df_selected.index)

        for col in metric_config.get("feature_cols", []):
            if pd.api.types.is_numeric_dtype(df_selected[col]):
                embedded_data[col] = pd.to_numeric(df_selected[col], errors="coerce")
                continue

            encoder = LabelEncoder()
            embedded_data[col] = encoder.fit_transform(df_selected[col].astype(str))

        embedded_data[group_col] = df_selected[group_col].values

        feature_columns = [col for col in embedded_data.columns if col != group_col]
        if not feature_columns:
            return MetricResult(
                description=f"MMD undefined: no comparable features for {group_col}",
                value=None,
                cluster="Consistency",
                threshold=None,
            )

        # normalize embedded data except group column
        for col in feature_columns:
            std = embedded_data[col].std()
            if pd.isna(std) or std == 0:
                continue
            embedded_data[col] = (embedded_data[col] - embedded_data[col].mean()) / std

        embedded_data = embedded_data.groupby(group_col).mean(numeric_only=True)

        missing_groups = [
            value for value in group_values if value not in embedded_data.index
        ]
        if missing_groups:
            return MetricResult(
                description=(
                    f"MMD undefined: missing comparison group(s) {missing_groups} for {group_col}"
                ),
                value=None,
                cluster="Consistency",
                threshold=None,
            )

        group_a, group_b = group_values
        distance = np.linalg.norm(
            embedded_data.loc[group_a] - embedded_data.loc[group_b]
        )

        return MetricResult(
            description=f"MMD",
            value=float(distance),
            cluster="Consistency",
            threshold=0.00001,
        )
