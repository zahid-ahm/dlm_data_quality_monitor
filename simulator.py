import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

def make_batch(day, fail=None):
    n = 50000 * (1 + 0.002 * day)
    if day % 7 in (5, 6):
        n *= 0.7
    if day >= 240:
        n *= 0.8  # source system migrated on day 240
    n = int(n * rng.normal(1, 0.03))
    if fail == 'loss':
        n = int(n * 0.9)
    if fail == 'dupes':
        n = int(n * 1.6)
    amount = rng.lognormal(3, 1, n)
    if fail == 'unit_change':
        amount = amount * 100
    p = 0.05 if fail == 'null_creep' else 0.02
    region = np.where(rng.random(n) < p, None, 'UK')
    return pd.DataFrame({'amount': amount, 'region': region})
