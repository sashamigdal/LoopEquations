"""Data-driven Re -> infinity extrapolation of the measured index, then theory fit.
Each run: u'^2 = S2(inf)/2, R(r) = 1 - S2/(2u'^2), integral scale L = Int_0^{r0} R dr (r0 = first zero of R).
alpha_exp(x), x = log(r/L), binned on a common grid, without the boundary-effects region r > W (W = common width of
finite_box_W2.py: a separation longer than the width of the flow cannot probe isotropic turbulence); at each x fit alpha = alpha_inf(x) + c(x) Re^-beta over mid+high Re runs.
Theory: alpha_inf(x) ~ alpha_D(x - s_inf)  (one parameter; optionally with a box k_min)."""
import json, numpy as np
from scipy.optimize import minimize_scalar, minimize
from finite_box import alphaL, load
from common import cache, res, load_json, save_json, strip, mpi_files, num, ALPHA_FIT
files = mpi_files()
P = load_json(res('mpi_physical_units.json')); Fb = load_json(cache('mpi_fit_box_full.json'))
W = load_json(res('mpi_fit_width_sub.json'))['W_sub']      # m
runs = []
for f, p, fb in zip(files, P, Fb):
    d = np.genfromtxt(f, delimiter=',', skip_header=1); rn, S2 = d[:, 1], d[:, 2]
    u2 = S2[-5:].mean()/2; R = 1-S2/(2*u2)
    i0 = np.argmax(R <= 0) if (R <= 0).any() else len(R)-1
    r = np.r_[0.0, rn[:i0+1]]; Rr = np.r_[1.0, R[:i0+1]]
    if R[i0] < 0:  # cut at the linear zero crossing
        t = Rr[-2]/(Rr[-2]-Rr[-1]); r[-1] = r[-2]+t*(r[-1]-r[-2]); Rr[-1] = 0.0
    L = np.trapezoid(Rr, r)               # in eta units
    lr, a, _ = load(f)
    runs.append(dict(Re=num(r'Re_([0-9.]+)_', f), L_eta=L, lr=lr, a=a, s=fb['s_inf'], lD_over_L=np.exp(fb['s_inf'])/L,
                     lr_W=float(np.log(W/p['eta'])), L_over_W=float(L*p['eta']/W)))
    print('Re=%5.0f  L/eta=%.3g  (L=%.3f m = %.2f W)   l_D/L from the tail fit = %.2f   points with r > W = %.2f m: %d' % (
        runs[-1]['Re'], L, L*p['eta'], L*p['eta']/W, runs[-1]['lD_over_L'], W, (lr > runs[-1]['lr_W']).sum()))
xg = np.arange(-4.0, 2.61, 0.1)
def binned(run):
    x = run['lr']-np.log(run['L_eta']); out = np.full(len(xg), np.nan)
    for j, x0 in enumerate(xg):
        m = (np.abs(x-x0) <= 0.05) & (run['lr'] <= run['lr_W'])      # boundary-effects region r > W excluded
        if m.sum() >= 1: out[j] = run['a'][m].mean()
    return out
Ab = np.array([binned(r) for r in runs]); Re = np.array([r['Re'] for r in runs])
fits = {}
for name, sel in [('mid+high (Re>=1046)', Re >= 1000), ('high (Re>=3070)', Re >= 3000)]:
    for beta in (0.5, 1.0):
        ainf = np.full(len(xg), np.nan); aerr = np.full(len(xg), np.nan)
        for j in range(len(xg)):
            y = Ab[sel, j]; X = Re[sel]**-beta; ok = np.isfinite(y)
            if ok.sum() >= 4:
                A = np.vstack([np.ones(ok.sum()), X[ok]]).T; c, rs, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
                ainf[j] = c[0]
                dof = ok.sum()-2; s2 = (rs[0]/dof) if (len(rs) and dof > 0) else np.nan
                cov = s2*np.linalg.inv(A.T@A) if np.isfinite(s2) else np.full((2, 2), np.nan); aerr[j] = np.sqrt(cov[0, 0])
        # theory fit to the extrapolated tail (alpha_inf < ALPHA_FIT, beyond the plateau)
        m = np.isfinite(ainf) & np.isfinite(aerr) & (aerr > 0) & (ainf < ALPHA_FIT) & (xg > -3.0)
        wt = 1/aerr[m]**2; nfit = int(m.sum())
        e1 = lambda s: np.sum(wt*(ainf[m]-alphaL(xg[m]-s, 0.0)[0])**2)
        r1 = minimize_scalar(e1, bounds=(-4, 4), method='bounded')
        e2 = lambda v: np.sum(wt*(ainf[m]-alphaL(xg[m]-v[0], abs(v[1]))[0])**2)
        r2 = min((minimize(e2, [r1.x, k0], method='Nelder-Mead') for k0 in (0.3, 1.0, 2.0, 3.0)), key=lambda z: z.fun)
        rmsw = lambda chi: np.sqrt(chi/wt.sum())            # weighted rms of alpha
        fits[(name, beta)] = dict(x=xg.tolist(), ainf=ainf.tolist(), aerr=aerr.tolist(), s=r1.x, rms=rmsw(r1.fun), chi2=r1.fun/(nfit-1),
                                 s2=r2.x[0], kmin=abs(r2.x[1]), rms2=rmsw(r2.fun), chi2_2=r2.fun/(nfit-2), npts=nfit)
        print('%-20s beta=%.1f : infinite system s=%.3f  chi2/dof=%.2f  w-rms=%.4f | box s=%.3f k_min=%.3f chi2/dof=%.2f w-rms=%.4f  (n=%d)' % (
            name, beta, r1.x, r1.fun/(nfit-1), rmsw(r1.fun), r2.x[0], abs(r2.x[1]), r2.fun/(nfit-2), rmsw(r2.fun), nfit))
out = {'runs': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in runs],
       'extrap': {f'{k[0]}|{k[1]}': v for k, v in fits.items()}}
save_json(out, cache('mpi_extrapolate_Re_full.json'))
save_json({'runs': strip(out['runs']), 'extrap': out['extrap']}, res('mpi_extrapolate_Re_inf.json'))
