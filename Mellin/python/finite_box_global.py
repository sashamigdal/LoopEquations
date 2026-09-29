"""Global finite-box test: one physical size for all runs, L/eta = Lam * r_max/eta (r_max = the largest measured separation),
k_min(theory units) = pi e^{s} / (Lam r_max/eta).  Per run only the shift s is fitted."""
import json, numpy as np
from scipy.optimize import minimize_scalar
from finite_box import alphaL
from common import cache, res, load_json
F = load_json(cache('mpi_fit_box_full.json'))
def fit_run(f, Lam):
    lr, a, i0 = np.array(f['lr']), np.array(f['a']), f['i0']; X, A = lr[i0:], a[i0:]; rmax = np.exp(lr[-1])
    e = lambda s: np.mean((A-alphaL(X-s, np.pi*np.exp(s)/(Lam*rmax))[0])**2)
    r = minimize_scalar(e, bounds=(f['s_inf']-1.5, f['s_inf']+1.5), method='bounded', options={'xatol': 1e-5})
    return r.x, r.fun, len(X)
tot0 = sum(f['rms_inf']**2*f['n'] for f in F); N = sum(f['n'] for f in F)
print('no box (11 params): pooled rms = %.4f' % np.sqrt(tot0/N))
grid = []
for Lam in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]:
    fits = [fit_run(f, Lam) for f in F]
    tot = sum(m*n for _, m, n in fits); grid.append((Lam, np.sqrt(tot/N), [x for x, _, _ in fits], [np.sqrt(m) for _, m, _ in fits]))
    print('Lambda=%.2f  pooled rms = %.4f   per-run rms: %s' % (Lam, np.sqrt(tot/N), ' '.join('%.3f' % np.sqrt(m) for _, m, _ in fits)))
best = min(grid, key=lambda t: t[1])
print('best Lambda = %.2f, pooled rms %.4f (free: 11 shifts + 1 Lambda) vs %.4f without box; free per-run kmin (22 params): %.4f' % (
    best[0], best[1], np.sqrt(tot0/N), np.sqrt(sum(f['rms_box']**2*f['n'] for f in F)/N)))
json.dump({'grid': [(l, r) for l, r, _, _ in grid], 'best': {'Lambda': best[0], 'rms': best[1], 'shifts': best[2], 'rms_runs': best[3]}},
          open(res('mpi_fit_box_global.json'), 'w'), indent=1)
