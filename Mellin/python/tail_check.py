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
