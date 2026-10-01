"""Robustness of the adopted wind-tunnel fit (extrap_invRe.py): alpha(x, Re) = alpha_inf(x) + c(x)/Re_lambda at fixed
x = log(r/L) over the runs Re_lambda >= 1046, then alpha_D(x - s) fitted to alpha_inf < ALPHA_FIT (weights 1/err^2).
  1. the level of the tail cut: ALPHA_FIT = 0.355 (adopted), 0.30, 0.25, 0.20, 0.15;
  2. leave-one-out: each run of the adopted set left out in turn;
  3. bootstrap over the runs (resampled with replacement, 2000 samples): the spread of s.
Input: cache/mpi_extrapolate_Re_full.json (extrapolate_Re.py; it holds per-run data and is not distributed).
Writes ../results/mpi_fit_robustness.json."""
import numpy as np
from scipy.optimize import minimize_scalar
from finite_box import alphaL
from common import cache, res, load_json, save_json, ALPHA_FIT

J = load_json(cache('mpi_extrapolate_Re_full.json')); runs = J['runs']
xg = np.arange(-4.0, 2.61, 0.1)


def binned(run):
    lr = np.array(run['lr']); x = lr-np.log(run['L_eta']); a = np.array(run['a']); out = np.full(len(xg), np.nan)
    for j, x0 in enumerate(xg):
        m = (np.abs(x-x0) <= 0.05) & (lr <= run['lr_W'])
        if m.any(): out[j] = a[m].mean()
    return out


Ab = np.array([binned(r) for r in runs]); Re = np.array([r['Re'] for r in runs])


def extrap(idx, minruns=4):
    ai = np.full(len(xg), np.nan); ae = np.full(len(xg), np.nan)
    for j in range(len(xg)):
        y = Ab[idx, j]; X = 1/Re[idx]; ok = np.isfinite(y)
        if ok.sum() < minruns or len(np.unique(X[ok])) < 3: continue
        A = np.vstack([np.ones(ok.sum()), X[ok]]).T; c, rs, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        dof = ok.sum()-2
        if not len(rs) or dof <= 0: continue
        ai[j] = c[0]; ae[j] = np.sqrt(rs[0]/dof*np.linalg.inv(A.T@A)[0, 0])
    return ai, ae


def fit(ai, ae, cut=ALPHA_FIT):
    m = np.isfinite(ai) & np.isfinite(ae) & (ae > 0) & (ai < cut) & (xg > -3.0)
    if m.sum() < 3: return np.nan, np.nan, np.nan, int(m.sum())
    w = 1/ae[m]**2; chi = lambda s: np.sum(w*(ai[m]-alphaL(xg[m]-s, 0.0)[0])**2)
    r = minimize_scalar(chi, bounds=(0, 4), method='bounded', options={'xatol': 1e-6})
    red = r.fun/(m.sum()-1); h = 1e-3; curv = (chi(r.x+h)-2*r.fun+chi(r.x-h))/h**2
    return r.x, np.sqrt(2/curv*max(red, 1.0)), red, int(m.sum())


if __name__ == '__main__':
    sel = np.where(Re >= 1000)[0]
    ai, ae = extrap(sel)
    out = {'adopted_runs': Re[sel].tolist()}
    s0, ds0, red0, n0 = fit(ai, ae)
    print('adopted (Re >= 1046, %d runs, cut alpha < %.3f): s = %.3f +- %.3f, chi2/dof = %.2f, %d bins' % (len(sel), ALPHA_FIT, s0, ds0, red0, n0))
    out['cut'] = []
    for cut in (0.355, 0.30, 0.25, 0.20, 0.15):
        s, ds, red, n = fit(ai, ae, cut)
        out['cut'].append(dict(cut=cut, s=s, ds=ds, chi2dof=red, bins=n))
        print('  tail cut alpha < %.3f: s = %.3f +- %.3f  chi2/dof = %.2f  (%d bins)   s - s_adopted = %+.3f' % (cut, s, ds, red, n, s-s0))
    out['leave_one_out'] = []
    for k in sel:
        s, ds, red, n = fit(*extrap(sel[sel != k]))
        out['leave_one_out'].append(dict(left_out=float(Re[k]), s=s, ds=ds, chi2dof=red, bins=n))
        print('  without Re = %5.0f: s = %.3f +- %.3f  chi2/dof = %.2f  (%d bins)   %+.3f' % (Re[k], s, ds, red, n, s-s0))
    lo = np.array([d['s'] for d in out['leave_one_out']])
    jack = np.sqrt((len(lo)-1)/len(lo)*np.sum((lo-lo.mean())**2))
    rng = np.random.default_rng(2026); bs = []
    for _ in range(2000):
        idx = rng.choice(sel, len(sel), replace=True)
        if len(np.unique(Re[idx])) < 4: continue
        s, ds, red, n = fit(*extrap(idx))
        if np.isfinite(s) and n >= 0.8*n0: bs.append(s)
    bs = np.array(bs)
    out['jackknife_ds'] = float(jack)
    out['bootstrap'] = dict(samples=int(len(bs)), mean=float(bs.mean()), std=float(bs.std()),
                            p16=float(np.percentile(bs, 16)), p84=float(np.percentile(bs, 84)))
    print('leave-one-out range %.3f .. %.3f, jackknife error %.3f; bootstrap over runs (%d samples): s = %.3f +- %.3f '
          '(16-84%%: %.3f .. %.3f); quoted fit error %.3f' % (lo.min(), lo.max(), jack, len(bs), bs.mean(), bs.std(),
          out['bootstrap']['p16'], out['bootstrap']['p84'], ds0))
    out['adopted'] = dict(s=s0, ds=ds0, chi2dof=red0, bins=n0)
    save_json(out, res('mpi_fit_robustness.json'), indent=1)
