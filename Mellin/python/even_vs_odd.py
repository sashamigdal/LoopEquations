"""alpha_D for the even ensemble (no prime-2 factor) vs odd, and their fits to the Re->inf extrapolated MPI tail."""
import json, numpy as np
from scipy.optimize import minimize_scalar
from common import cache, res, load_json, ALPHA_FIT
line = np.load(cache('lineD_c1.npy')); Y = line[0].real; Zodd = line[1]; Q = 1.0+1j*Y; h = Y[1]-Y[0]
Zeven = Zodd*(1-2.0**(Q-7.5))                       # M_even = M_odd (1 - 2^{-(p+17/2)}),  p = -1-q
wS = np.full(len(Y), 2.0); wS[1:-1:2] = 4.0; wS[0] = wS[-1] = 1.0; wS *= h/3/np.pi
def alpha(xi, Z):
    E = np.exp(np.outer(np.atleast_1d(xi), Q))*Z; D = (E@wS).real; D1 = ((E*Q)@wS).real; return D1/D
J = load_json(res('mpi_extrap_invRe.json')); x = np.array(J['x']); A = J['fits']['Re>=1046 (adopted)']
ai, ae = np.array(A['ainf'], float), np.array(A['aerr'], float)
m = np.isfinite(ai) & np.isfinite(ae) & (ae > 0) & (ai < ALPHA_FIT) & (x > -3.0); w = 1/ae[m]**2
out = {}
for name, Z in [('odd', Zodd), ('even', Zeven)]:
    chi = lambda s: np.sum(w*(ai[m]-alpha(x[m]-s, Z))**2)
    r = minimize_scalar(chi, bounds=(0, 4), method='bounded', options={'xatol': 1e-6})
    out[name] = (r.x, r.fun/(m.sum()-1))
    print('%-4s ensemble: s = %.4f  chi2/dof = %.3f' % (name, r.x, r.fun/(m.sum()-1)))
xx = np.linspace(-3, 3, 13)
print('max |alpha_odd - alpha_even| on log r in [-3, 3] at equal xi:', np.max(np.abs(alpha(xx, Zodd)-alpha(xx, Zeven))))
np.save(cache('alpha_even.npy'), np.vstack([np.arange(-8, 8.001, 0.01), alpha(np.arange(-8, 8.001, 0.01), Zeven)]))
