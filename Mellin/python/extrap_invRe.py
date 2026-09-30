"""Adopted: alpha(x, Re) = alpha_inf(x) + c(x)/Re_lambda at fixed x = log(r/L); theory alpha_D(x - s) fitted to alpha_inf (weights 1/err^2).
Robustness vs the set of runs, uncertainty of s, linearity in 1/Re."""
import json, numpy as np
from scipy.optimize import minimize_scalar
from finite_box import alphaL
from common import cache, res, load_json, save_json, ALPHA_FIT
J = load_json(cache('mpi_extrapolate_Re_full.json')); runs = J['runs']
xg = np.arange(-4.0, 2.61, 0.1)
def binned(run):
    x = np.array(run['lr'])-np.log(run['L_eta']); a = np.array(run['a']); out = np.full(len(xg), np.nan)
    for j, x0 in enumerate(xg):
        m = np.abs(x-x0) <= 0.05
        if m.any(): out[j] = a[m].mean()
    return out
Ab = np.array([binned(r) for r in runs]); Re = np.array([r['Re'] for r in runs])
def extrap(sel, minruns=4):
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
full = {'x': xg.tolist(), 'Re': Re.tolist(), 'binned': np.where(np.isfinite(Ab), Ab, None).tolist(), 'fits': out}
save_json(full, cache('mpi_extrap_invRe_full.json'))
save_json({k: v for k, v in full.items() if k != 'binned'}, res('mpi_extrap_invRe.json'))   # alpha_inf(x) with errors, no per-run data
