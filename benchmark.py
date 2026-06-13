import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import simulator
from simulator import make_batch
from profiler import profile
from dlm import DLM
from monitor import Monitor

def run_monitor(seed, fail_day):
    simulator.rng = np.random.default_rng(seed)
    fails = {d: 'loss' for d in range(fail_day, fail_day + 14)}
    vals = np.array([np.log(profile(make_batch(d))['rows']) for d in range(28)])
    wd = np.arange(28) % 7
    res = vals - np.array([vals[wd == d].mean() for d in wd])
    model = DLM(vals.mean(), res.var())
    mon = Monitor()
    lo, hi = vals.mean() - 3 * vals.std(), vals.mean() + 3 * vals.std()
    out = []
    for day in range(28, 300):
        y = np.log(profile(make_batch(day, fails.get(day)))['rows'])
        _, _, f, S = model.predict()
        alert, p, B = mon.check(y, f, S)
        model.update(y)
        if alert:
            model.intervene(model.P[0, 0] * 20)
        out.append((day, y, f, np.sqrt(S), alert, not lo < y < hi))
    return out

delays, sdelays, fa, sfa = [], [], 0, 0
for i in range(40):
    fail_day = 150 + (i * 7) % 100
    out = run_monitor(i, fail_day)
    pre = [r for r in out if r[0] < fail_day]
    post = [r for r in out if fail_day <= r[0] < fail_day + 14]
    fa += sum(r[4] for r in pre)
    sfa += sum(r[5] for r in pre)
    d = [r[0] - fail_day for r in post if r[4]]
    sd = [r[0] - fail_day for r in post if r[5]]
    delays.append(d[0] if d else 14)
    sdelays.append(sd[0] if sd else 14)

print('                       dlm monitor   static 3sd')
print(f'mean detection delay   {np.mean(delays):11.1f}   {np.mean(sdelays):10.1f}')
print(f'missed (of 40)         {sum(d == 14 for d in delays):11d}   {sum(d == 14 for d in sdelays):10d}')
print(f'false alarms pre-fail  {fa:11d}   {sfa:10d}')

out = [r for r in run_monitor(0, 180) if r[0] < 230]
day, y, f, sd, alert, _ = map(np.array, zip(*out))
plt.figure(figsize=(11, 4))
plt.fill_between(day, f - 2 * sd, f + 2 * sd, alpha=0.3, label='forecast ± 2sd')
plt.plot(day, y, lw=0.8, label='log daily rows')
plt.plot(day[alert], y[alert], 'rv', label='alert')
plt.axvspan(180, 194, color='grey', alpha=0.2, label='silent 10% record loss')
plt.legend()
plt.tight_layout()
plt.savefig('figures/rows.png', dpi=150)
