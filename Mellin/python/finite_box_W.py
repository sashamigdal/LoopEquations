"""One PHYSICAL width W (metres) for all runs.  Theory variable rho = |r1-r2|/l_D, l_D = sqrt(nu~ (t+t0)) = eta e^s  (s fitted per run),
so the cutoff in theory units is k_min = pi l_D / W = pi eta e^s / W.  Also: per-run profile of rms vs k_min (how well k_min is determined)."""
import json, numpy as np
from scipy.optimize import minimize_scalar
from finite_box import alphaL
from common import cache, res, load_json
F = load_json(cache('mpi_fit_box_full.json')); P = load_json(res('mpi_physical_units.json'))
def tail(f): lr, a, i0 = np.array(f['lr']), np.array(f['a']), f['i0']; return lr[i0:], a[i0:]
def fit_s(f, eta, W):
    X, A = tail(f)
    e = lambda s: np.mean((A-alphaL(X-s, np.pi*eta*np.exp(s)/W)[0])**2)
    r = minimize_scalar(e, bounds=(f['s_inf']-1.5, f['s_inf']+1.5), method='bounded', options={'xatol': 1e-5})
    return r.x, r.fun, len(X)
if __name__ == '__main__':
    N = sum(f['n'] for f in F); rms_inf = np.sqrt(sum(f['rms_inf']**2*f['n'] for f in F)/N); rms_free = np.sqrt(sum(f['rms_box']**2*f['n'] for f in F)/N)
    grid = []
    for W in [1.0, 1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 5.0, 8.0]:
        fits = [fit_s(f, p['eta'], W) for f, p in zip(F, P)]
        pooled = np.sqrt(sum(m*n for _, m, n in fits)/N); grid.append((W, pooled, fits))
        print('W = %4.1f m  pooled rms = %.4f   per run: %s' % (W, pooled, ' '.join('%.3f' % np.sqrt(m) for _, m, _ in fits)))
    Wb, rb, fb = min(grid, key=lambda t: t[1])
    # refine
    r = minimize_scalar(lambda W: np.sqrt(sum(fit_s(f, p['eta'], W)[1]*f['n'] for f, p in zip(F, P))/N), bounds=(1.5, 4.0), method='bounded', options={'xatol': 1e-3})
    Wb = r.x; fb = [fit_s(f, p['eta'], Wb) for f, p in zip(F, P)]
    print('\nbest W = %.3f m : pooled rms %.4f  (infinite system %.4f with 11 params; free k_min per run %.4f with 22)' % (Wb, r.fun, rms_inf, rms_free))
    for f, p, (s, m, n) in zip(F, P, fb):
        print('  Re=%5.0f  s=%.3f  l_D=%.3f m  k_min=%.3f  rms %.4f (inf %.4f, free %.4f)' % (f['Re'], s, p['eta']*np.exp(s), np.pi*p['eta']*np.exp(s)/Wb, np.sqrt(m), f['rms_inf'], f['rms_box']))
    # profile of rms vs k_min per run (s re-fitted at each k_min): how sharply is k_min determined?
    print('\nprofile rms(k_min) per run (s refitted):')
    kgrid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    prof = {}
    for f in F:
        X, A = tail(f); row = []
        for km in kgrid:
            e = lambda s: np.mean((A-alphaL(X-s, km)[0])**2)
            rr = minimize_scalar(e, bounds=(f['s_inf']-1.5, f['s_inf']+1.5), method='bounded'); row.append(float(np.sqrt(rr.fun)))
        prof[f['Re']] = row; print('  Re=%5.0f  ' % f['Re'] + ' '.join('%.3f' % v for v in row))
    json.dump({'W_grid': [(W, p) for W, p, _ in grid], 'W_best': Wb, 'rms_best': r.fun, 'rms_inf': rms_inf, 'rms_free': rms_free,
               'runs': [dict(Re=f['Re'], s=s, lD_m=p['eta']*np.exp(s), kmin=np.pi*p['eta']*np.exp(s)/Wb, rms=np.sqrt(m)) for f, p, (s, m, n) in zip(F, P, fb)],
               'profile_kgrid': kgrid, 'profile': prof}, open(res('mpi_fit_width.json'), 'w'), indent=1)
