import numpy as np
from simulator import make_batch
from profiler import profile
from dlm import DLM
from monitor import Monitor

fails = {d: 'loss' for d in range(120, 130)}
fails[160] = 'dupes'
fails.update({d: 'null_creep' for d in range(180, 200)})
fails[220] = 'unit_change'
MIGRATION = 240  # known in advance, tell the model instead of alarming

def transform(metrics):
    return {'rows': np.log(metrics['rows']),
            'null_rate': metrics['null_rate'],
            'mean_amount': metrics['mean_amount']}

warmup = [transform(profile(make_batch(d))) for d in range(28)]
models, monitors, skipped = {}, {}, {}
for name in warmup[0]:
    vals = np.array([w[name] for w in warmup])
    wd = np.arange(28) % 7
    res = vals - np.array([vals[wd == d].mean() for d in wd])
    models[name] = DLM(np.mean(vals), np.var(res))
    monitors[name] = Monitor()
    skipped[name] = False

for day in range(28, 365):
    if day == MIGRATION:
        for name in models:
            models[name].intervene(models[name].P[0, 0] * 50)
            monitors[name].C = 1.0
        print(f'day {day}  -- planned migration, prior variance inflated --')
    obs = transform(profile(make_batch(day, fails.get(day))))
    line = f'day {day} '
    for name in models:
        y = obs[name]
        line += f' {name}={y:.3f}'
        _, _, f, S = models[name].predict()
        alert, p, B = monitors[name].check(y, f, S)
        if alert:
            print(f'\nALERT day {day}  {name}')
            print(f'  observed {y:.4f}  expected {f:.4f}  sd {np.sqrt(S):.4f}')
            print(f'  tail prob {p:.2g}  bayes factor {B:.1f}\n')
        if p < 1e-8 and not skipped[name]:
            models[name].update(None)  # lone wild outlier, don't pollute the state
            skipped[name] = True
        else:
            models[name].update(y)  # two in a row is a shift, not an outlier
            skipped[name] = False
            if alert:
                models[name].intervene(models[name].P[0, 0] * 20)  # let it re-adapt
    print(line)
