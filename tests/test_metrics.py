import pytest
import pandas as pd

from metric.metric import Metric, MetricResult
from metric.metrics.representativeness import Range, IQR, StdDev, HillNumbers


def test_metric_registry():
    class TestMetric(Metric):
        def compute(self, data, reference=None, metric_config=None):
            return MetricResult(description="test", value=1.0)

    assert "TestMetric" in Metric.registry
    metric_instance = Metric.registry["TestMetric"]()
    result = metric_instance.compute(pd.DataFrame())
    assert isinstance(result, MetricResult)
    assert result.description == "test"
    assert result.value == 1.0


def test_metric_abstract_method():
    with pytest.raises(TypeError):

        class IncompleteMetric(Metric):
            pass

        IncompleteMetric()


def test_range_result():
    metric_instance = Range()
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 8]})
    result_a = metric_instance.compute(df, metric_config={"column": "a"})
    assert isinstance(result_a, MetricResult)
    assert result_a.description == "a range"
    assert result_a.value == 2

    result_b = metric_instance.compute(df, metric_config={"column": "b"})
    assert isinstance(result_b, MetricResult)
    assert result_b.description == "b range"
    assert result_b.value == 4


def test_IQR_result():
    metric_instance = IQR()
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [4, 5, 6, 7, 10]})
    result_a = metric_instance.compute(df, metric_config={"column": "a"})
    assert isinstance(result_a, MetricResult)
    assert result_a.description == "a IQR"
    assert result_a.value == 2.0
    result_b = metric_instance.compute(df, metric_config={"column": "b"})
    assert isinstance(result_b, MetricResult)
    assert result_b.description == "b IQR"
    assert result_b.value == 2.0


def test_StdDev_result():
    metric_instance = StdDev()
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [4, 5, 6, 7, 8]})
    result_a = metric_instance.compute(df, metric_config={"column": "a"})
    assert isinstance(result_a, MetricResult)
    assert result_a.description == "a StdDev"
    assert result_a.value == pytest.approx(1.5811388300841898)
    result_b = metric_instance.compute(df, metric_config={"column": "b"})
    assert isinstance(result_b, MetricResult)
    assert result_b.description == "b StdDev"
    assert result_b.value == pytest.approx(1.5811388300841898)


def test_HillNumber_result():
    metric_instance = HillNumbers()
    df = pd.DataFrame({"a": [1, 1, 1, 1, 2, 2, 2, 2]})
    metric_config = {"column": "a", "types": [1, 2]}
    result = metric_instance.compute(df, metric_config=metric_config)
    assert isinstance(result, MetricResult)
    assert result.description.startswith("Hill Numbers q=")
    assert result.value == pytest.approx(2.0)
