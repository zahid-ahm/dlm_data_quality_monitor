# Data Quality Monitoring with Dynamic Linear Models

Monitors daily pipeline metrics (row count, null rate, column mean) and flags when one deviates from expected behaviour.

Each metric is modelled with a Bayesian dynamic linear model that tracks its linear trend and weekly seasonality. The model forecasts each day, and a sequential likelihood ratio test flags a metric when it crosses a threshold.

## How to run

```
pip install -r requirements.txt
python run.py
```

The simulator generates a year of daily batches with DQ issues at scheduled points: a silent record loss, a duplicate load, a null-rate creep, a unit change, and a planned migration. 

`python benchmark.py` runs 40 simulations of a 14-day silent record loss against a static baseline (warmup mean plus or minus three sd):

| Metric | DLM monitor | Static threshold (3 sd) |
|---|---|---|
| Mean detection delay (days) | 1.5 | 14.0 |
| Missed (of 40) | 1 | 40 |
| False alarms before fault | 27 | 24 |

*14.0 = never detected within the 14-day window.*

The static baseline misses every loss. The band must be wide enough to contain the weekly cycle, so a 10% drop falls within it and goes undetected. Being fixed, it cannot follow the upward trend either, so normal days eventually cross its upper edge and raise false alarms. The DLM catches the loss in a day or two at a comparable false alarm rate.

![rows](figures/rows.png)

## Limitations

- Gaussian noise with constant variance holds well for log row count but is weaker for near-zero rates.
- Weekly seasonality only. Monthly and holiday effects need extra components.
- Metrics are monitored independently. Pooling correlated failures would improve detection.