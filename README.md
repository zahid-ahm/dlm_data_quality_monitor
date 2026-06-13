# Data Quality Monitoring with Dynamic Linear Models

## Overview

The application monitors daily data quality metrics from a batch pipeline and detects when any of them departs from expected behaviour. Three are tracked: row count, null rate, and column mean.

Every metric is modelled by a Bayesian dynamic linear model that captures its trend and weekly seasonality. Each day the model forecasts the value one step ahead. The realised value is then compared against this forecast, and the discrepancy is assessed through a sequential likelihood ratio test that accumulates evidence across successive days. When that evidence exceeds a fixed threshold, an alert is raised.

Several cases beyond the scope of a static threshold are handled directly. Planned changes are communicated in advance and absorbed without an alarm. An isolated extreme value is flagged but excluded from the state. A day with no load is treated through the standard missing-observation step. For evaluation, a simulator generates a year of pipeline data with faults inserted at known points, and a benchmark compares the monitor against a conventional static-threshold baseline.

## How to run

```
pip install -r requirements.txt
python run.py
```

The simulator generates one year of daily batches with faults introduced at scheduled points: a silent record loss, a duplicate load, a gradual rise in null rate, a unit change, and a planned migration. Alerts are printed to stdout:

```
ALERT day 120  rows
  observed 11.0716  expected 11.1759  sd 0.0290
  tail prob 0.00033  bayes factor 1364.6
```

`python benchmark.py` runs 40 simulations of a 14-day silent record loss and compares the monitor against a static baseline, set at the warmup mean plus or minus three standard deviations:

```
                       dlm monitor   static 3sd
mean detection delay           1.5         14.0
missed (of 40)                   1           40
false alarms pre-fail           27           24
```

The static baseline fails to detect any of the losses. Its band must be wide enough to contain the weekly cycle, so a 10% decline falls within it and goes unnoticed. The upward trend, meanwhile, carries ordinary values past the upper limit and produces false alarms regardless. The dynamic linear model detects the loss within a day or two at a comparable false alarm rate, approximately one alert per 250 clean metric-days.

![rows](figures/rows.png)

## Limitations

- Observation noise is assumed Gaussian with constant variance. This holds reasonably for the log row count, but less so for raw counts or rates near zero. A proper count likelihood would require simulation-based inference rather than closed-form updates, which remains future work.
- Seasonality is restricted to the weekly cycle. Monthly and holiday effects can be incorporated by adding further components.
- The metrics are monitored independently. A genuine failure typically perturbs several at once, and pooling that evidence would improve sensitivity.