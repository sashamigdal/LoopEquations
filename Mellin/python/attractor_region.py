"""Turbulent attractor vs stochastization stage in the Max Planck runs (Sec. III D of the paper).

Each run is a separate flow measured at one station, so the scaling test is done across runs at fixed alpha:
r_alpha(run) is the separation where the measured index crosses the level alpha.
  * attractor region:     alpha(r) is one function of r / l_run, with a single length per run, so the shape
                          log(r_alpha / r_ref) is the same for all runs (r_ref = r at alpha = 0.3);
  * stochastization stage: r_alpha scales with the Kolmogorov length eta instead (d log(r_alpha/eta)/d log Re ~ 0).
The attractor boundary is the largest alpha for which the scatter of log(r_alpha/r_ref) across the 11 runs stays below 0.1.
Writes ../results/mpi_attractor_region.json (statistics only) and cache/attractor_crossings.json (per-run crossings)."""
import numpy as np
from common import cache, res, load_json, save_json, ALPHA_ATTR, ALPHA_FIT

LEVELS = [1.2, 1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
REF, TOL = 0.3, 0.1


def crossing(x, a, lev):
    """first downward crossing of the level, scanning from small r; linear interpolation in x"""
    for i in range(len(a)-1):
        if a[i] >= lev > a[i+1]:
            return x[i] + (a[i]-lev)*(x[i+1]-x[i])/(a[i]-a[i+1])
    return np.nan


def slope(y, lre):
    ok = np.isfinite(y)
    A = np.vstack([np.ones(ok.sum()), lre[ok]]).T
    c, rs, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    err = np.sqrt(rs[0]/(ok.sum()-2)*np.linalg.inv(A.T@A)[1, 1])
    return float(c[1]), float(err)


if __name__ == '__main__':
    runs = load_json(cache('mpi_extrapolate_Re_full.json'))['runs']          # lr = log(r/eta), a = alpha, L_eta = L/eta
    P = load_json(res('mpi_physical_units.json'))
    Re = np.array([r['Re'] for r in runs]); lre = np.log(Re)
    Xeta = np.array([[crossing(np.array(r['lr']), np.array(r['a']), lev) for lev in LEVELS] for r in runs])   # log(r_alpha/eta)
    XL = Xeta - np.log([r['L_eta'] for r in runs])[:, None]                                                  # log(r_alpha/L)
    shape = Xeta - Xeta[:, [LEVELS.index(REF)]]                                                              # log(r_alpha/r_ref)
    out = {'levels': LEVELS, 'ref_level': REF, 'tolerance': TOL, 'Re': Re.tolist(), 'rows': []}
    print('alpha   scatter of log(r_alpha/r_%.1f)   d log(r_alpha/eta)/d log Re   d log(r_alpha/L)/d log Re' % REF)
    for k, lev in enumerate(LEVELS):
        sc_all, sc_hi = float(np.nanstd(shape[:, k])), float(np.nanstd(shape[Re >= 1000, k]))
        be, bee = slope(Xeta[:, k], lre); bl, ble = slope(XL[:, k], lre)
        out['rows'].append(dict(alpha=lev, scatter_all=sc_all, scatter_Re1046=sc_hi, slope_eta=be, slope_eta_err=bee, slope_L=bl, slope_L_err=ble))
        print('%5.2f   %.3f (Re>=1046: %.3f)          %+.3f +- %.3f                 %+.3f +- %.3f' % (lev, sc_all, sc_hi, be, bee, bl, ble))
    ok = [row['alpha'] for row in out['rows'] if row['scatter_all'] <= TOL]
    a_star = max(ok)
    out['alpha_attractor'] = a_star
    tau = [p['u_rms']**2/p['eps'] for p in P]
    out['tau_s'] = [min(tau), max(tau)]
    print('\nattractor region (scatter <= %.1f): alpha <= %.2f   [common.ALPHA_ATTR = %.2f]; theory fitted to alpha < %.2f'
          % (TOL, a_star, ALPHA_ATTR, ALPHA_FIT))
    print('decay time tau = u\'^2/eps at the station: %.2f - %.2f s for all runs (L/eta varies %.0f - %.0f)'
          % (min(tau), max(tau), min(r['L_eta'] for r in runs), max(r['L_eta'] for r in runs)))
    save_json(out, res('mpi_attractor_region.json'), indent=1)
    save_json({'Re': Re.tolist(), 'levels': LEVELS, 'log_r_over_eta': np.where(np.isfinite(Xeta), Xeta, None).tolist(),
               'runs': [{'Re': r['Re'], 'lr': r['lr'], 'a': r['a'], 'L_eta': r['L_eta']} for r in runs]}, cache('attractor_crossings.json'))
