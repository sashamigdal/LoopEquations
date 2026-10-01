"""Lefschetz thimble of the energy spectrum (odd ensemble), the p-plane problem of MellinOdd.nb and Ref. [migdal2026Riemann, S7]:

    H(xi) = (1/2 pi i) Int_{eps-i inf}^{eps+i inf} M(p) e^{p xi} dp,   xi = log kappa,  -7/2 < eps < 0,
    M(p)  = f(p) Gamma(-p) zeta(p+15/2) / ((2p+7)(2p+17) zeta(p+17/2) (1 - 2^-(p+17/2))).

W(p) = log M(p) + p xi; real saddle p0 in (-7/2, 0); thimble W(p(t)) = W(p0) - t^2, p'(t) = -2t/W'(p), p(-t) = conj p(t);
Gauss-Hermite with the exact-integrand factor g(t) = e^{W(p(t)) - W(p0) + t^2} p'(t), as for D(rho) (thimble.py):
    H_thimble = (e^{W0}/pi) sum_{t_j>0} w_j Im g(t_j),   dH/dxi uses the same nodes with an extra factor p.
The walls lie to the LEFT of the Mellin contour:
    Riemann  P_n = -8 + i gamma_n            (zeros of zeta(p+17/2)), paired with the zeros Z_n = -7 + i gamma_n,
    dyadic   D_m = -17/2 + 2 pi i m / log 2  (zeros of 1 - 2^-(p+17/2)), m >= 1.
A pole is trapped between the Mellin line and the upper thimble when the LEFTWARD ray {Im p = Im P, Re p < Re P} crosses the
thimble an odd number of times. A thimble that ends on a zero Z_n (zero parking) is continued straight up Re p = -7, to the
right of both walls, which adds the tail (1/pi) Int_{gamma_n}^inf Re[M e^{p xi}] dy and no crossings. Then
    H = H_thimble + tail + 2 Re sum_{trapped} Res_{p=P} M(p) e^{p xi}.
zeta is evaluated by Euler-Maclaurin (fastzeta.py).

Status: work in progress for the revision of Figs. 5-6 (fine Stokes tracking, Berry smoothing); not yet validated against
lineH.py and not used by run_all.sh or the paper."""
import numpy as np, mpmath as mm
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
from scipy import special as sp
from dfcore import DFfast
import fastzeta as fz
from common import cache
import os

DF = DFfast(nGL=200)
L2 = np.log(2.0)
NR, ND = 120, 30


def _zeros(n):
    f = cache('zetazeros_%d.npy' % n)
    if os.path.exists(f):
        return np.load(f)
    mm.mp.dps = 20
    g = np.array([float(mm.zetazero(k).imag) for k in range(1, n+1)])
    np.save(f, g)
    return g


GAMMA = _zeros(NR)                                   # gamma_n
DYAD = 2*np.pi*np.arange(1, ND+1)/L2                 # 2 pi m / log 2


def logM(p):
    p = complex(p); u = np.exp(-(p+8.5)*L2)
    return (sp.loggamma(-p) + np.log(fz.zeta(p+7.5)) - np.log(fz.zeta(p+8.5)) - np.log(2*p+7) - np.log(2*p+17)
            + np.log(complex(DF(p))) - np.log(1-u))


def dlogM(p):
    """W'(p) - xi"""
    p = complex(p); u = np.exp(-(p+8.5)*L2)
    return (-sp.psi(-p) + fz.logder(p+7.5) - fz.logder(p+8.5) - 2/(2*p+7) - 2/(2*p+17)
            + complex(DF(p, 1)/DF(p, 0)) - L2*u/(1-u))


def d2logM(p):
    p = complex(p); u = np.exp(-(p+8.5)*L2); d0, d1, d2 = DF(p, 0), DF(p, 1), DF(p, 2)
    return (complex(mm.psi(1, -mm.mpc(p))) + fz.logder2(p+7.5) - fz.logder2(p+8.5) + 4/(2*p+7)**2 + 4/(2*p+17)**2
            + complex(d2/d0-(d1/d0)**2) + L2**2*u/(1-u)**2)


def saddle(xi):
    return brentq(lambda x: dlogM(x).real + xi, -3.5+1e-9, -1e-9, xtol=1e-15, rtol=1e-15, maxiter=200)


def xi_of_p0(p0):
    return -dlogM(p0).real


def residue(kind, k, xi):
    """Res_{p=P} M(p) e^{p xi} and the pole P, for the Riemann (kind 'R', gamma_k) or dyadic (kind 'D', m = k+1) pole"""
    if kind == 'R':
        p = complex(-8, GAMMA[k]); s = p+8.5
        z0, z1, _ = fz.zeta_d(s)
        den = (2*p+7)*(2*p+17)*z1*(1-np.exp(-s*L2))
    else:
        p = complex(-8.5, DYAD[k]); s = p+8.5
        den = (2*p+7)*(2*p+17)*fz.zeta(s)*L2
    num = np.exp(sp.loggamma(-p))*fz.zeta(p+7.5)*complex(DF(p))
    return num/den*np.exp(p*xi), p


class ThimbleH:
    def __init__(self, xi, nGH=80, eps=1e-3, rtol=1e-11, atol=1e-13):
        self.xi = xi
        self.p0 = p0 = saddle(xi)
        self.W2 = d2logM(p0).real
        self.v0 = 1j*np.sqrt(2/self.W2)
        h = 1e-4
        W3 = ((d2logM(p0+h)-d2logM(p0-h))/(2*h)).real
        self.c2 = W3/(3*self.W2**2)
        x, w = hermgauss(nGH)
        self.tn, self.wn = x[x > 1e-14], w[x > 1e-14]
        self.w0 = w[np.abs(x) <= 1e-14].sum()
        tmax = self.tn.max()+0.25
        rhs = lambda t, y: [-2*t/(dlogM(y[0])+xi)]
        ev = lambda t, y: 600-abs(y[0]); ev.terminal = True
        y0 = p0 + self.v0*eps + self.c2*eps**2
        self.sol = solve_ivp(rhs, (eps, tmax), [complex(y0)], method='DOP853', dense_output=True, rtol=rtol, atol=atol, events=ev)
        self.eps, self.tend = eps, self.sol.t[-1]
        self.path = self.sol.y[0]
        self.W0 = logM(p0).real + xi*p0

    def q(self, t):
        t = np.atleast_1d(t); out = np.empty(len(t), complex); small = t < self.eps
        out[small] = self.p0 + self.v0*t[small] + self.c2*t[small]**2
        if (~small).any():
            out[~small] = self.sol.sol(t[~small])[0]
        return out

    def evaluate(self):
        xi = self.xi; ok = self.tn <= self.tend; t = self.tn[ok]; w = self.wn[ok]; ps = self.q(t)
        g = np.array([np.exp(logM(pi)+xi*pi-self.W0+ti**2)*(-2*ti/(dlogM(pi)+xi)) for ti, pi in zip(t, ps)])
        T = np.exp(self.W0)/np.pi*((w*g.imag).sum() + self.w0*self.v0.imag/2)
        T1 = np.exp(self.W0)/np.pi*((w*(ps*g).imag).sum() + self.w0*self.p0*self.v0.imag/2)
        return T, T1, int((~ok).sum())

    def dense(self, n=20000):
        t = np.linspace(self.eps, self.tend, n)
        return np.concatenate([[self.p0], self.q(t)])

    def terminal_zero(self, tol=1e-3):
        e = self.path[-1]
        k = int(np.argmin(np.abs(e-(-7+1j*GAMMA))))
        return k if abs(e-(-7+1j*GAMMA[k])) < tol else None

    def trapped(self, P=None):
        """(trapped Riemann indices, trapped dyadic indices): odd number of crossings of the leftward rays"""
        P = self.dense() if P is None else P
        x, y = P.real, P.imag
        z = self.terminal_zero()
        d = P[-1]-P[-50]
        out = []
        for heights, wall in ((GAMMA, -8.0), (DYAD, -8.5)):
            idx = []
            for n, h in enumerate(heights):
                s = (y[:-1]-h)*(y[1:]-h); cnt = 0
                for k in np.where(s <= 0)[0]:
                    if y[k+1] == y[k]: continue
                    xc = x[k]+(h-y[k])*(x[k+1]-x[k])/(y[k+1]-y[k]); cnt += xc < wall
                if z is None and h > y[-1] and d.imag > 0:          # unfinished path: continue with its final direction
                    cnt += (x[-1]+(h-y[-1])*d.real/d.imag) < wall
                if cnt % 2: idx.append(n)
            out.append(idx)
        return out[0], out[1], z

    def tail(self, z):
        """continuation from the parking zero Z = -7 + i gamma_z up Re p = -7: (1/pi) Int Re[M e^{p xi}] dy, and with a factor p"""
        if z is None:
            return 0.0, 0.0
        xi, g0 = self.xi, GAMMA[z]
        F = lambda y, m: (np.exp(logM(complex(-7, y))+xi*complex(-7, y))*complex(-7, y)**m).real
        a = quad(F, g0, g0+80, args=(0,), limit=400, epsabs=0, epsrel=1e-12)[0]/np.pi
        b = quad(F, g0, g0+80, args=(1,), limit=400, epsabs=0, epsrel=1e-12)[0]/np.pi
        return a, b


def compute(xi, nGH=80):
    th = ThimbleH(xi, nGH)
    T, T1, miss = th.evaluate()
    iR, iD, z = th.trapped()
    ta, tb = th.tail(z)
    SR = SR1 = SD = SD1 = 0.0
    for k in iR:
        r, p = residue('R', k, xi); SR += 2*r.real; SR1 += 2*(p*r).real
    for k in iD:
        r, p = residue('D', k, xi); SD += 2*r.real; SD1 += 2*(p*r).real
    H = T+ta+SR+SD; H1 = T1+tb+SR1+SD1
    return dict(xi=xi, p0=th.p0, T=T, T1=T1, tail=ta, tail1=tb, SR=SR, SR1=SR1, SD=SD, SD1=SD1, H=H, H1=H1, n=H1/H,
                iR=iR, iD=iD, zero=z, missing=miss, end=[th.path[-1].real, th.path[-1].imag], tend=th.tend)
