"""Data-driven Re -> infinity extrapolation of the measured index, then theory fit.
Each run: u'^2 = S2(inf)/2, R(r) = 1 - S2/(2u'^2), integral scale L = Int_0^{r0} R dr (r0 = first zero of R).
alpha_exp(x), x = log(r/L), binned on a common grid; at each x fit alpha = alpha_inf(x) + c(x) Re^-beta over mid+high Re runs.
Theory: alpha_inf(x) ~ alpha_D(x - s_inf)  (one parameter; optionally with a box k_min)."""
import json, glob, re, numpy as np
from scipy.optimize import minimize_scalar, minimize
from finite_box import alphaL, load
num = lambda pat, f: float(re.search(pat, f.split('/')[-1]).group(1).rstrip('.'))
files = sorted(glob.glob('../mpi/E_Kohler/Re_*.csv'), key=lambda f: num(r'Re_([0-9.]+)_', f))
P = json.load(open('../mpi/physical_units.json')); Fb = json.load(open('../mpi/fit_box.json'))
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
    runs.append(dict(Re=num(r'Re_([0-9.]+)_', f), L_eta=L, lr=lr, a=a, s=fb['s_inf'], lD_over_L=np.exp(fb['s_inf'])/L))
    print('Re=%5.0f  L/eta=%.3g  (L=%.3f m)   l_D/L from the tail fit = %.2f' % (runs[-1]['Re'], L, L*p['eta'], runs[-1]['lD_over_L']))
xg = np.arange(-4.0, 2.61, 0.1)
def binned(run):
    x = run['lr']-np.log(run['L_eta']); out = np.full(len(xg), np.nan)
    for j, x0 in enumerate(xg):
        m = np.abs(x-x0) <= 0.05
        if m.sum() >= 1: out[j] = run['a'][m].mean()
    return out
Ab = np.array([binned(r) for r in runs]); Re = np.array([r['Re'] for r in runs])
res = {}
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
        # theory fit to the extrapolated tail (alpha_inf < 0.355, beyond the plateau)
        m = np.isfinite(ainf) & np.isfinite(aerr) & (aerr > 0) & (ainf < 0.355) & (xg > -3.0)
        wt = 1/aerr[m]**2; nfit = int(m.sum())
        e1 = lambda s: np.sum(wt*(ainf[m]-alphaL(xg[m]-s, 0.0)[0])**2)
        r1 = minimize_scalar(e1, bounds=(-4, 4), method='bounded')
        e2 = lambda v: np.sum(wt*(ainf[m]-alphaL(xg[m]-v[0], abs(v[1]))[0])**2)
        r2 = min((minimize(e2, [r1.x, k0], method='Nelder-Mead') for k0 in (0.3, 1.0, 2.0, 3.0)), key=lambda z: z.fun)
        rmsw = lambda chi: np.sqrt(chi/wt.sum())            # weighted rms of alpha
        res[(name, beta)] = dict(x=xg.tolist(), ainf=ainf.tolist(), aerr=aerr.tolist(), s=r1.x, rms=rmsw(r1.fun), chi2=r1.fun/(nfit-1),
                                 s2=r2.x[0], kmin=abs(r2.x[1]), rms2=rmsw(r2.fun), chi2_2=r2.fun/(nfit-2), npts=nfit)
        print('%-20s beta=%.1f : infinite system s=%.3f  chi2/dof=%.2f  w-rms=%.4f | box s=%.3f k_min=%.3f chi2/dof=%.2f w-rms=%.4f  (n=%d)' % (
            name, beta, r1.x, r1.fun/(nfit-1), rmsw(r1.fun), r2.x[0], abs(r2.x[1]), r2.fun/(nfit-2), rmsw(r2.fun), nfit))
json.dump({'runs': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in runs],
           'extrap': {f'{k[0]}|{k[1]}': v for k, v in res.items()}}, open('../mpi/extrapolate_Re.json', 'w'))
