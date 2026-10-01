"""Enstrophy decay of the freely decaying turbulence of Matsuzawa, Zhu, Goldenfeld and Irvine, PNAS 123, e2526858123 (2026),
from their published data (Zenodo, doi:10.5281/zenodo.18380405; not distributed here). Set MATSUZAWA_DATA_DIR to the folder
with Fig3A_dissipation_rate.h5 (blob) and Fig2A_energy_decay.h5 (double oscillating grid).

eps = -dq/dt = nu <omega^2>; hypotheses
    A: eps ~ (t - t0)^(-9/4)   (Euler ensemble, E ~ t^-5/4)
    B: eps ~ (t - t0)^(-11/5)  (Saffman, E ~ t^-6/5)
    (C: (t - t0)^(-17/7), Loitsyansky, which needs an initial spectrum tuned to cancel the k^2 term: for reference only)
on the widest straight piece of log eps vs log(t - t0): the widest window whose straight-line fit has an rms residual within
K = 2 times the local noise of the data (residuals of local quadratic fits). In the window ln eps = c - n ln(t - t0); each
hypothesis fixes n, the intercept is free, and P(B)/P(A) = exp(-[chi2_B - chi2_A]/2), chi2 with sigma = rms residual of the
free fit. The residuals are correlated (eps is a smoothed time derivative): the chi2 differences are divided by the
integrated autocorrelation time tau of the residuals (effective number of independent points N/tau).
Writes ../results/matsuzawa_enstrophy.json."""
import glob, os, numpy as np, h5py
from common import res, save_json
D = os.environ.get('MATSUZAWA_DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Matsuzawa'))
HYP = {'A (9/4, Euler ensemble)': 9/4, 'B (11/5, Saffman)': 11/5, 'C (17/7, Loitsyansky)': 17/7}


def _file(name):
    f = glob.glob(os.path.join(D, '*'+name))
    if not f: raise SystemExit('Matsuzawa et al. data (%s) not found in %s' % (name, D))
    return f[0]


def blob():
    with h5py.File(_file('Fig3A_dissipation_rate.h5')) as f:
        eps, tt = f['dqdt'][()], f['t_t0'][()]                      # dqdt is stored as -dq/dt
    ok = np.isfinite(eps) & (eps > 0) & (tt > 0); o = np.argsort(tt[ok])
    return tt[ok][o], eps[ok][o]


def grid():
    with h5py.File(_file('Fig2A_energy_decay.h5')) as f:
        g = f['doubgle_oscillating_grid']; t, q, t0 = g['t'][()], g['q'][()], g['t0'][()]
    o = np.argsort(t); t, q = t[o]-t0, q[o]
    ok = (t > 0) & np.isfinite(q) & (q > 0); t, q = t[ok], q[ok]
    lt = np.log(t)
    tm = np.exp(0.5*(lt[1:]+lt[:-1])); e = -(q[1:]-q[:-1])/(t[1:]-t[:-1])
    ok = e > 0
    return tm[ok], e[ok]


def noise(lt, le, half=0.1):
    r = []
    for i in range(len(lt)):
        m = np.abs(lt-lt[i]) <= half
        if m.sum() >= 6:
            c = np.polyfit(lt[m], le[m], 2); r.append(le[i]-np.polyval(c, lt[i]))
    return np.std(r)


def tau_int(r):
    r = r-r.mean(); c0 = np.dot(r, r); t = 1.0
    for k in range(1, len(r)//2):
        rho = np.dot(r[:-k], r[k:])/c0
        if rho <= 0: break
        t += 2*rho
    return t


def widest(lt, le, thr, step=0.01, minpts=12):
    """widest window [a, a+W] in log10 t whose linear-fit rms <= thr"""
    best = None
    for W in np.arange(0.3, lt.max()-lt.min(), 0.02):
        found = None
        for a in np.arange(lt.min(), lt.max()-W+1e-9, step):
            m = (lt >= a) & (lt <= a+W)
            if m.sum() < minpts: continue
            p = np.polyfit(lt[m], le[m], 1); rms = (le[m]-np.polyval(p, lt[m])).std()
            if rms <= thr and (found is None or rms < found[0]): found = (rms, a, W)
        if found: best = found
    return best


def test(name, t, e, K, half=0.1, minpts=12):
    lt10, le = np.log10(t), np.log(e); sig0 = noise(lt10, le, half)
    rms, a, W = widest(lt10, le, K*sig0, minpts=minpts)
    m = (lt10 >= a) & (lt10 <= a+W); x = np.log(t[m]); y = le[m]; N = m.sum()
    A = np.vstack([np.ones(N), -x]).T; c, *_ = np.linalg.lstsq(A, y, rcond=None); r = y-A@c
    sig = r.std(ddof=2); tau = tau_int(r)
    cov = sig**2*np.linalg.inv(A.T@A); dn = np.sqrt(cov[1, 1])
    chi = {}
    for h, n in HYP.items():
        cX = np.mean(y+n*x); chi[h] = np.sum((y-cX+n*x)**2)/sig**2
    hA, hB, hC = list(HYP)
    d = chi[hB]-chi[hA]
    print('%s, K = %.1f: noise %.3f; widest straight piece t - t0 = %.2f .. %.2f s (%.2f decades, %d points), rms %.3f'
          % (name, K, sig0, 10**a, 10**(a+W), W, N, rms))
    print('   free slope n = %.3f +- %.3f (white noise), +- %.3f with tau = %.1f points (N_eff = %.0f)'
          % (c[1], dn, dn*np.sqrt(tau), tau, N/tau))
    for h in HYP:
        print('   %-24s chi2 - chi2_free = %7.1f  (%.1f with tau)' % (h, chi[h]-(N-2), (chi[h]-(N-2))/tau))
    print('   P(B)/P(A) = exp(-%.1f/2) = %.2g ;  with correlated noise exp(-%.2f/2) = %.2g' % (d, np.exp(-d/2), d/tau, np.exp(-d/2/tau)))
    print('   P(C)/P(A) = %.2g ;  with correlated noise %.2g' % (np.exp(-(chi[hC]-chi[hA])/2), np.exp(-(chi[hC]-chi[hA])/2/tau)))
    return dict(t1=10**a, t2=10**(a+W), n=c[1], dn=dn, tau=tau, N=N, ratioBA=np.exp(-d/2), ratioBA_corr=np.exp(-d/2/tau),
                ratioCA_corr=np.exp(-(chi[hC]-chi[hA])/2/tau), m=m, c=c, rms=rms)


if __name__ == '__main__':
    out = {}
    for name, data, half, minpts in (('blob', blob(), 0.1, 12), ('grid', grid(), 0.4, 6)):   # the grid has 31 points
        for K in (1.5, 2.0, 3.0):
            r = test(name, *data, K, half, minpts)
            out['%s K=%.1f' % (name, K)] = {k: float(v) for k, v in r.items() if k not in ('m', 'c')}
    save_json(out, res('matsuzawa_enstrophy.json'), indent=1)
