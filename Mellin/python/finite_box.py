"""Finite-box structure function  D_L(r) = 2 Int_{kmin}^inf (1 - sin kr/(kr)) H(k) dk = D_inf(r) - dD(r),
dD(r) = 2 Int_0^{kmin} (1 - sin kr/(kr)) H(k) dk  (H(k) from its exact entire small-k series),
alpha_L = r d log D_L/dr.  Fit (s, kmin) to each Max Planck run's tail."""
import json, numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize, minimize_scalar
from hseries import Hser
from common import cache, res, save_json, strip, mpi_files, mpi_index, num
line = np.load(cache('lineD_c1.npy')); Y = line[0].real; Z = line[1]; Q = 1.0+1j*Y; h = Y[1]-Y[0]
wS = np.full(len(Y), 2.0); wS[1:-1:2] = 4.0; wS[0] = wS[-1] = 1.0; wS *= h/3/np.pi      # Simpson weights incl. 1/pi
def Dinf(xi):
    """exact D_inf and r dD_inf/dr at xi = log r (== thimble result to 5e-10)"""
    E = np.exp(np.outer(np.atleast_1d(xi), Q))*Z
    return (E@wS).real, ((E*Q)@wS).real
def dD(r, kmin):
    r = np.atleast_1d(r)
    if kmin <= 0: return np.zeros_like(r), np.zeros_like(r)
    n = int(80 + 2.0*kmin*r.max()); t, w = leggauss(n)
    k = kmin*(t+1)/2; w = w*kmin/2*Hser(k)
    x = np.outer(r, k)
    s = np.sin(x)/x
    return 2*((1-s)@w), 2*((-np.cos(x)+s)@w)
def alphaL(xi, kmin):
    D, D1 = Dinf(xi); d0, d1 = dD(np.exp(xi), kmin)
    return (D1-d1)/(D-d0), D-d0
load = mpi_index
def fit_run(f):
    """per-run fits of the tail: shift s alone (infinite system), and (s, kappa_min) (finite width)"""
    Re = num(r'Re_([0-9.]+)_', f); lr, a, i0 = load(f); X, A = lr[i0:], a[i0:]
    e1 = lambda s: np.mean((A-alphaL(X-s, 0.0)[0])**2)
    r1 = minimize_scalar(e1, bounds=(5, 16), method='bounded')
    e2 = lambda v: np.mean((A-alphaL(X-v[0], abs(v[1]))[0])**2)
    best = None
    for k0 in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        rr = minimize(e2, [r1.x, k0], method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-10, 'maxiter': 2000})
        if best is None or rr.fun < best.fun: best = rr
    s2, km = best.x[0], abs(best.x[1])
    return dict(file=f.split('/')[-1], Re=Re, n=int(len(X)), s_inf=float(r1.x), rms_inf=float(np.sqrt(r1.fun)),
                s_box=float(s2), kmin=float(km), rms_box=float(np.sqrt(best.fun)),
                L_theory=float(np.pi/km), L_over_eta=float(np.pi/km*np.exp(s2)), rmax_over_eta=float(np.exp(lr[-1])),
                lr=lr.tolist(), a=a.tolist(), i0=int(i0))


if __name__ == '__main__':
    from multiprocessing import Pool
    # sanity: kmin = 0.1 reproduces the old xi2 of CorrelationOscillation.nb (k in [0.1, 1000]); xi2(1) there = 0.05625510
    print('alpha_L(r=1, kmin=0.1) =', alphaL(np.array([0.0]), 0.1)[0][0], ' (old notebook xi2[1] = 0.0562551028, with htab)')
    with Pool(4) as p:
        out = p.map(fit_run, mpi_files())          # the runs are independent; results do not depend on the pool size
    for r in out:
        lr = np.array(r['lr']); s2, km = r['s_box'], np.float64(r['kmin'])
        print('Re=%5.0f  no-box: s=%.3f rms=%.4f | box: s=%.3f kmin=%.3f (L_th=%.2f) rms=%.4f  L/eta=%.3g  r_max/eta=%.3g  ratio L/r_max=%.2f' % (
            r['Re'], r['s_inf'], r['rms_inf'], s2, km, np.pi/km, r['rms_box'], np.pi/km*np.exp(s2), np.exp(lr[-1]), np.pi/km*np.exp(s2)/np.exp(lr[-1])))
    save_json(out, cache('mpi_fit_box_full.json')); save_json(strip(out), res('mpi_fit_box.json'), indent=1)
