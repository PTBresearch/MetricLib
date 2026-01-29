# MetricLib Project

## Overview
`MetricLib`: an extensible toolkit for holistic data quality evaluation of medical ML datasets, based on the theoretical METRIC-framework for trustworthy AI in medicine. The toolkit is able to process a range of data modalities in a memory-efficient manner. While a core set of DQ metrics is implemented, `MetricLib` is easily extensible with custom metrics and therefore allows investigation of use-case-specific requirements. Additionally, by aggregated DQ scores, the tool enables the efficient identification of data quality gaps.

## Project Structure
```
metriclib/
    __init__.py
    data.py
    metric.py
    report.py
    metrics/
        measurement_process.py
        timeliness.py
        representativeness.py
        informativeness.py
        consistency.py
notebooks/
    example_ptbxl.ipynb
data/
tests/
    __init__.py
    test_dataset.py
    test_metrics.py
    test_report.py
```

## Installation
1. Clone the repository:
   ```bash
   git clone git@gitlab1.ptb.de:martin.seyferth/metriclib.git
   cd metriclib
   ```
2. Install locally:
   ```bash
   pip install -e .
   ```

## Usage
An example implementation of creating a Data Quality report can be found [here](notebooks/example_ptbxl.ipynb)

## Metrics
|        Metric Name        | Implemented | Tested |
|---------------------------|-------------|--------|
| Hill Numbers              |      x      |    x   |
| Mean                      |      x      |    x   |
| Standard deviation        |      x      |    x   |
| IQR                       |      x      |    x   |
| Syntactic Consistency     |      x      |        |
| Limit of Quantification   |      x      |        |
| Sample Entropy            |      x      |        |
| Maximum Mean Discrepency  |      x      |        |



A documentation of required and optional parameters can be found here: https://gitlab1.ptb.de/martin.seyferth/metric/-/tree/main/metric/metrics

## Testing
Run the unit tests using pytest:
```bash
pytest tests/
```