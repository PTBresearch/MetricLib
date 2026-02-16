import numpy as np
import antropy as ant
from ..metric import MetricResult, StreamMetric, TabularMetric


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


class SNR(StreamMetric):
    def aggregate(self, datapoint, reference=None, metric_config=None):
        if reference is None:
            raise ValueError("Reference signal is required for SNR calculation.")

        signal_power = np.mean(datapoint[0] ** 2)
        noise_power = np.mean((datapoint[0] - reference[0]) ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def compute(self, data, reference, metric_config):
        return MetricResult(
            cluster=None,
            threshold=0,
            description="Mean SNR across all leads",
            value=np.array(data).mean(),
        )


class MetadataCompleteness(TabularMetric):
    def compute(self, data, reference=None, metric_config=None):
        # count missing values in relation to all cells in the DataFrame
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)

        return MetricResult(
            description="Metadata Completeness",
            value=completeness,
            cluster="Representativeness",
            threshold=1.0,
        )
