"""First comparison (README figures 6, 7): each Max Planck run's tail alpha_exp < 0.355 fitted by the infinite-system alpha_D(log(r/eta) - s)."""
import numpy as np
from scipy.optimize import minimize_scalar
from common import cache, res, save_json, strip, mpi_files, mpi_index, num
o = np.load(cache('oscill.npy')); d = (o[0], o[1], o[2])    # exact alpha_D(xi) on a 0.01 grid
aD = lambda x: np.interp(x, d[0], d[2])
out = []
for f in mpi_files():
    Re = num(r'Re_([0-9.]+)_', f); eps = num(r'Eps_([0-9.]+)\.csv', f)
    lr, a, i0 = mpi_index(f)       # tail window as in CorrelationOscillation.nb
    X, A = lr[i0:], a[i0:]
    err = lambda s: np.mean((A-aD(X-s))**2)
    r = minimize_scalar(err, bounds=(0, 20), method='bounded')
    s = r.x; rms = np.sqrt(r.fun); resid = A-aD(X-s)
    out.append(dict(lr=lr.tolist(), a=a.tolist(), i0=int(i0), file=f.split('/')[-1], Re=Re, eps=eps, shift=s, rms=rms, n=int(len(X)),
                    logr_range=[float(X[0]), float(X[-1])]))
    print('Re_lambda=%6.0f  shift ln(r/eta)-xi = %6.3f (log10 %.3f)  rms=%.4f  n=%d  |resid|max=%.3f' % (Re, s, s/np.log(10), rms, len(X), np.abs(resid).max()))
save_json(out, cache('mpi_fit_tail_full.json')); save_json(strip(out), res('mpi_fit.json'), indent=1)
R = np.array([o['Re'] for o in out]); S = np.array([o['shift'] for o in out])
p = np.polyfit(np.log(R), S, 1); print('shift vs ln Re_lambda: slope %.3f (L/eta ~ Re_lambda^1.5 -> 1.5), intercept %.3f' % tuple(p))
