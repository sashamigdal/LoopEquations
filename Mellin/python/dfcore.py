"""Fast DF(p,n) = Int_{D1}^{D2} (1-x) d^n/dp^n [20 C^(p-1) (A C - B p)] dx for complex p.
ABC(x) from Chebyshev table (smooth), inner x-integral by Gauss-Legendre (inner quadrature only)."""
import json, numpy as np
from numpy.polynomial import chebyshev as Ch, legendre as Le

def load(path='abc_cheb48.json'):
    d = json.load(open(path))
    rows = np.array([[float(v) for v in r] for r in d['rows']])
    D1, D2 = float(d['D1']), float(d['D2'])
    x, r, L, S, J, IQ = rows.T
    Q = np.exp(IQ/(8*np.pi))
    A = Q*2*(r-6)/(r+12); B = Q*J/S; C = Q*L/(2*np.pi*S)
    u = (2*x - (D1+D2))/(D2-D1)
    deg = len(x)-1
    fits = {k: Ch.chebfit(u, v, deg) for k, v in (('A', A), ('B', B), ('logC', np.log(C)), ('IQ', IQ), ('r', r))}
    return D1, D2, fits

class DFfast:
    def __init__(self, path='abc_cheb48.json', nGL=128):
        self.D1, self.D2, self.fits = load(path)
        t, w = Le.leggauss(nGL)
        self.x = (self.D1+self.D2)/2 + (self.D2-self.D1)/2*t
        self.w = w*(self.D2-self.D1)/2*(1-self.x)*20
        u = t
        self.A = Ch.chebval(u, self.fits['A']); self.B = Ch.chebval(u, self.fits['B'])
        self.lC = Ch.chebval(u, self.fits['logC']); self.C = np.exp(self.lC)
    def abc(self, x):
        u = (2*x - (self.D1+self.D2))/(self.D2-self.D1)
        return Ch.chebval(u, self.fits['A']), Ch.chebval(u, self.fits['B']), np.exp(Ch.chebval(u, self.fits['logC']))
    def __call__(self, p, n=0):
        p = np.asarray(p, dtype=complex)[..., None]
        A, B, C, lC = self.A, self.B, self.C, self.lC
        base = np.exp((p-1)*lC)
        g = A*C - B*p
        if n == 0: z = base*g
        elif n == 1: z = base*(-B + g*lC)
        else: z = base*lC*(-2*B + g*lC)
        return (z*self.w).sum(-1)
