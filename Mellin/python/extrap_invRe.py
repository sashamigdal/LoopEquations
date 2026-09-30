"""Adopted: alpha(x, Re) = alpha_inf(x) + c(x)/Re_lambda at fixed x = log(r/L); theory alpha_D(x - s) fitted to alpha_inf (weights 1/err^2).
Robustness vs the set of runs, uncertainty of s, linearity in 1/Re.
The boundary-effects region r > W of each run is left out (extrapolate_Re.py); cache keeps it separately for the figure."""
import json, numpy as np
from scipy.optimize import minimize_scalar
from finite_box import alphaL
from common import cache, res, load_json, save_json, ALPHA_FIT
J = load_json(cache('mpi_extrapolate_Re_full.json')); runs = J['runs']
xg = np.arange(-4.0, 2.61, 0.1)
def binned(run, boundary=False, keep_all=False):
    """bin means at fixed x; boundary=True returns the boundary-effects region r > W only, keep_all=True all points"""
    lr = np.array(run['lr']); x = lr-np.log(run['L_eta']); a = np.array(run['a']); out = np.full(len(xg), np.nan)
    inW = np.ones(len(lr), bool) if keep_all else (lr > run['lr_W']) if boundary else (lr <= run['lr_W'])
    for j, x0 in enumerate(xg):
        m = (np.abs(x-x0) <= 0.05) & inW
        if m.any(): out[j] = a[m].mean()
    return out
Ab = np.array([binned(r) for r in runs]); Re = np.array([r['Re'] for r in runs])
def extrap(sel, minruns=4, Ab=Ab):
    ai = np.full(len(xg), np.nan); ae = np.full(len(xg), np.nan); sl = np.full(len(xg), np.nan); lin = np.full(len(xg), np.nan)
    for j in range(len(xg)):
        y = Ab[sel, j]; X = 1/Re[sel]; ok = np.isfinite(y)
        if ok.sum() < minruns: continue
        A = np.vstack([np.ones(ok.sum()), X[ok]]).T; c, rs, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        dof = ok.sum()-2; s2 = rs[0]/dof if len(rs) and dof > 0 else np.nan
        ai[j], sl[j] = c; ae[j] = np.sqrt(s2*np.linalg.inv(A.T@A)[0, 0]); lin[j] = np.sqrt(s2)
    return ai, ae, sl, lin
def fit(ai, ae):
    m = np.isfinite(ai) & np.isfinite(ae) & (ae > 0) & (ai < ALPHA_FIT) & (xg > -3.0)
    w = 1/ae[m]**2; chi = lambda s: np.sum(w*(ai[m]-alphaL(xg[m]-s, 0.0)[0])**2)
    r = minimize_scalar(chi, bounds=(0, 4), method='bounded', options={'xatol': 1e-6})
    n = m.sum(); red = r.fun/(n-1)
    h = 1e-3; curv = (chi(r.x+h)-2*r.fun+chi(r.x-h))/h**2          # d2chi/ds2
    ds = np.sqrt(2/curv*max(red, 1.0))                              # 1-sigma, inflated by chi2/dof
    return r.x, ds, red, np.sqrt(r.fun/w.sum()), n, m
out = {}
for name, lo in [('all 11 runs (Re>=413)', 0), ('Re>=1046 (adopted)', 1000), ('Re>=1305', 1300), ('Re>=2033', 2000), ('Re>=2398', 2390)]:
    sel = Re >= lo
    ai, ae, sl, lin = extrap(sel)
    s, ds, red, wr, n, m = fit(ai, ae)
    out[name] = dict(runs=int(sel.sum()), s=s, ds=ds, chi2dof=red, wrms=wr, n=int(n), ainf=ai.tolist(), aerr=ae.tolist(), slope=sl.tolist(), linres=lin.tolist())
    print('%-24s runs=%2d  s = %.3f +- %.3f  (l_D/L = %.2f)  chi2/dof = %.2f  weighted rms = %.4f  (n=%d bins)' % (name, sel.sum(), s, ds, np.exp(s), red, wr, n))
# check: the adopted fit with the boundary-effects region kept
Ak = np.array([binned(r, keep_all=True) for r in runs]); sel = Re >= 1000
ai, ae, sl, lin = extrap(sel, Ab=Ak); s, ds, red, wr, n, m = fit(ai, ae)
res_k = (ai-alphaL(xg-s, 0.0)[0])[m]; big = np.argsort(-np.abs(res_k))[:2]
out['Re>=1046, boundary region r > W kept'] = dict(runs=int(sel.sum()), s=s, ds=ds, chi2dof=red, wrms=wr, n=int(n),
                                                  largest_residuals=[[float(xg[m][i]), float(res_k[i])] for i in big])
print('%-24s runs=%2d  s = %.3f +- %.3f  chi2/dof = %.2f  weighted rms = %.4f  (n=%d bins); largest residuals %s' % (
    'r > W kept (check)', sel.sum(), s, ds, red, wr, n, ', '.join('%+.3f at log(r/L) = %.1f' % (res_k[i], xg[m][i]) for i in big)))
Bb = np.array([binned(r, boundary=True) for r in runs])
full = {'x': xg.tolist(), 'Re': Re.tolist(), 'binned': np.where(np.isfinite(Ab), Ab, None).tolist(),
        'binned_boundary': np.where(np.isfinite(Bb), Bb, None).tolist(), 'x_W': [r['lr_W']-np.log(r['L_eta']) for r in runs],
        'L_over_W': [r['L_over_W'] for r in runs], 'fits': out}
save_json(full, cache('mpi_extrap_invRe_full.json'))
save_json({k: v for k, v in full.items() if k not in ('binned', 'binned_boundary')}, res('mpi_extrap_invRe.json'))   # alpha_inf(x) with errors, no per-run data
