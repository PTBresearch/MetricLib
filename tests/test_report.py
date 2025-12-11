import pytest
import pandas as pd

from metric.report import Report
from metric.metric import Metric, MetricResult
from metric.data import Dataset


class TestDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return (None, "A", {"meta": "value"})


class TestMetric(Metric):
    def compute(self, data, reference=None, metric_config=None):
        return MetricResult(
            description="test",
            value=1.0,
            cluster="Measurement Process",
            threshold=1.0,
        )


def test_report_add_metric():
    dataset = TestDataset()
    report = Report(datasets=[dataset])
    report.add_metric("TestMetric")
    assert "TestMetric" in report.metrics


def test_report_generate():
    dataset = TestDataset()
    report = Report(datasets=[dataset])
    report.add_metric("TestMetric")
    report.generate()
    result = report.metrics["TestMetric"]["result"]
    assert isinstance(result, MetricResult)
    assert result.description == "test"
    assert result.value == 1.0
    assert report.scores["Measurement Process"] == 1.0


def test_available_metrics():
    available_metrics = list(Metric.registry.keys())
    assert "TestMetric" in available_metrics
