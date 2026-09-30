"""Numbers of Sec. III D (why only the tail is compared):
  measured vs theoretical index below r = L, the Reynolds trend at fixed r/L, and the effect of the finite width at small rho."""
import numpy as np
from finite_box import alphaL
from common import cache, res, load_json, have_mpi
J = load_json(res('mpi_extrap_invRe.json')); x = np.array(J['x']); A = J['fits']['Re>=1046 (adopted)']
ai, ae, s = np.array(A['ainf'], float), np.array(A['aerr'], float), A['s']
th = alphaL(x-s, 0.0)[0]
for x0 in (-4.0, -1.0, 0.2, 1.1):
    j = int(np.argmin(abs(x-x0))); print('log(r/L) = %4.1f: alpha_inf = %.3f +- %.3f, theory %.3f' % (x[j], ai[j], ae[j], th[j]))
xs = np.array([-6, -5, -4, -3, -2, -1, 0.0])
print('finite width: max |alpha_W - alpha| for kappa_min <= 3.6 at log rho =', xs, ':',
      np.round(np.max([np.abs(alphaL(xs, k)[0]-alphaL(xs, 0.0)[0]) for k in (0.6, 1, 2, 3, 3.6)], axis=0), 4))
if have_mpi():
    F = load_json(cache('mpi_extrap_invRe_full.json')); B = F['binned']; j = int(np.argmin(abs(x+4.0)))
    print('alpha at log(r/L) = -4 per run:', ', '.join('Re %d: %.2f' % (r, row[j]) for r, row in zip(F['Re'], B) if row[j] is not None))
# fitting the extrapolated index beyond the turbulent attractor (Sec. III D): chi2/dof for wider tail cuts
from scipy.optimize import minimize_scalar
for name in ('Re>=1046 (adopted)', 'Re>=2398'):
    F = J['fits'][name]; ai_, ae_ = np.array(F['ainf'], float), np.array(F['aerr'], float); row = []
    for cut in (0.355, 0.40, 0.45, 0.50):
        m = np.isfinite(ai_) & np.isfinite(ae_) & (ae_ > 0) & (ai_ < cut) & (x > -3.0); w = 1/ae_[m]**2
        chi = lambda s_: np.sum(w*(ai_[m]-alphaL(x[m]-s_, 0.0)[0])**2)
        r = minimize_scalar(chi, bounds=(0, 4), method='bounded', options={'xatol': 1e-6})
        row.append('alpha < %.3f: chi2/dof = %.2f' % (cut, r.fun/(m.sum()-1)))
    print('%-20s ' % name + ' | '.join(row))
