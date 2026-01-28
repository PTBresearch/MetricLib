import numpy as np
import antropy as ant
from ..metric import MetricResult, StreamMetric


class LimitofQuantification(StreamMetric):
    def aggregate(self, datapoint, reference=None, metric_config=None):
        if metric_config["cp"] is None:
            cp = 10
        else:
            cp = metric_config["cp"]

        if metric_config["LoB"] is None:
            raise ValueError("metric_config must include 'LoB' key")
        else:
            LoB = metric_config["LoB"]

        LoQ = LoB + cp * datapoint[0].mean()

        return LoQ

    def compute(self, data, reference, metric_config):
        return MetricResult(
            cluster=None,
            threshold=0,
            description="Proportion of values below limit of quantification",
            value=np.array(data).mean(),
        )


class SampleEntropy(StreamMetric):
    def aggregate(self, datapoint, reference=None, metric_config=None):
        values = []
        for i in range(datapoint[0].shape[0]):
            m = 2
            r = 0.2 * np.std(datapoint[0][i, :])
            values.append(ant.sample_entropy(datapoint[0][i, :], m, r))
        entropy = np.mean(values)

        return entropy

    def compute(self, data, reference, metric_config):
        return MetricResult(
            cluster=None,
            threshold=0,
            description="Mean sample entropy across all leads",
            value=np.array(data).mean(),
        )
