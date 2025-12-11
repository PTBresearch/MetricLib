from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
import pandas as pd
from typing import Any, List, Union, Tuple, TypeVar

from .data import Dataset


@dataclass
class MetricResult:
    description: str
    # A metric can return a single number or a list (e.g., per-column values)
    value: Any
    # Optional grouping and threshold info for scoring/aggregation
    cluster: Union[str, None] = None
    threshold: Union[float, None] = None


_T_co = TypeVar("_T_co", covariant=True)


class AggregationMetric(ABC):
    """
    Abstract base class for data point metrics.
    All data point metric classes should inherit from this class and implement the `compute` method.
    """

    registry = {}

    def store(fn):
        def wrapper(*args, **kwargs):
            self = args[0]
            result = fn(*args, **kwargs)
            self.result.append(result)

            return result

        return wrapper

    def __init__(self):
        self.result: List = []

    def __init_subclass__(cls):
        super().__init_subclass__()
        AggregationMetric.registry[cls.__name__] = cls

        agg = cls.__dict__.get("aggregate", None)

    @abstractmethod
    def aggregate(
        self,
        data_point: _T_co,
        reference: Union[_T_co, None] = None,
        metric_config: Union[str, None] = None,
    ) -> Tuple[float]:
        """Assess a single data point using this metric and return the results.

            Parameters
            - data_point: T_co
                    The data point that should be assessed by this metric. This is
                    the primary data point under inspection.

            - reference: Optional[T_co]
                    An optional, cleaned reference data point that can act as a
                    clean version of the data point. Metrics that need a canonical or
                    expected version of the data (for example correctness against a
                    known-good source) should accept and use this data point. If not
                    needed by a metric, `None` is allowed.

            - metric_config: Optional[str]
                    Optional path or JSON string containing metric-specific
                    configuration. Use this to keep the method signature compact;
                    all metric-specific parameters (thresholds, aggregation options,
                    etc.) can be stored here.

        Returns
        - MetricResult
            Tuple of floats representing the metric's output values.
        """
        raise NotImplementedError()

    def compute(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: Union[str, None] = None,
    ) -> MetricResult:
        """Aggregate the results collected from `add` calls and return the final metric result.

            Parameters
            - data: pd.DataFrame
                    The DataFrame that has been assessed by this metric through multiple
                    `add` calls.

            - reference: Optional[pd.DataFrame]
                    An optional, cleaned reference DataFrame that can act as a
                    clean version of the dataset. Metrics that need a canonical or
                    expected version of the data (for example correctness against a
                    known-good source) should accept and use this DataFrame. If not
                    needed by a metric, `None` is allowed.

            - metric_config: Optional[str]
                    Optional path or JSON string containing metric-specific
                    configuration. Use this to keep the method signature compact;
                    all metric-specific parameters (thresholds, aggregation options,
                    etc.) can be stored here.

        Returns
        - MetricResult
            Tuple of floats representing the aggregated metric's output values.
        """
        return MetricResult()


# Adapted from https://github.com/HPI-Information-Systems/Metis/blob/main/metis/metric/metric.py
class Metric(ABC):
    """
    Abstract base class for metrics.
    All metric classes should inherit from this class and implement the `compute` method.
    """

    registry = {}

    def __init_subclass__(cls):
        super().__init_subclass__()
        Metric.registry[cls.__name__] = cls

    @abstractmethod
    def compute(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: Union[str, None] = None,
    ) -> MetricResult:
        """Assess data using this metric and return the results as a MetricResult.

        Implementations must avoid mutating the input data and should parse
        metric_config as needed, raising clear exceptions for invalid configs.
        """
        raise NotImplementedError()
