"""Saddle point + Lefschetz thimble + Gauss-Hermite evaluation of

    I(r) = (1/2 pi i) Int_{c-i inf}^{c+i inf} r^q e^{S(q)} dq ,   xi = log r,

for the MellinOscillationsOdd problem, with
    ZR(q) = -(64 Sqrt2 Pi Csc[Pi q/2] DF(-1-q) Zeta[13/2-q]) / ((128 Sqrt2-2^q)(1+q)(2q-15)(2q-5) Zeta[15/2-q]).

  kind='f' : e^S = ZR,      strip -1 < Re q < 0   ->  f(r) = Int H(k) sin(kr)/(kr) dk
  kind='D' : e^S = -2 ZR,   strip  0 < Re q < 2   ->  D(r) = 2 (f(0) - f(r))

Both share S'(q), so they share the thimble ODE; only the saddle strip differs.
Thimble: W(q) = S(q) + xi q,  W(q(t)) = W0 - t^2,  q'(t) = -2 t / W'(q),  q(-t) = conj q(t).
Gauss-Hermite: I = (e^{W0}/pi) [ sum_{t_i>0} w_i Im g(t_i) (+ w_0 Im v0 / 2) ],
    g(t) = e^{W(q(t)) - W0 + t^2} q'(t)   (== q'(t) on an exact thimble; the factor makes the
    result exact by Cauchy even if the numerical path drifts off the thimble).
Stokes corrections: I_line = I_thimble - 2 Re sum_{trapped upper poles} Res  (+ tails of paths
    that terminate on a zero of the integrand).  alpha = dlog I/dxi uses the same nodes with an extra q.
"""
import numpy as np, mpmath as mm
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from dfcore import DFfast

mm.mp.dps = 20
DF = DFfast(nGL=200)
S2 = np.sqrt(2.0); K = 128*S2; L2 = np.log(2.0)
GAMMA = [float(mm.zetazero(n).imag) for n in range(1, 61)]            # Riemann wall 7 + i gamma_n
DYAD = [2*np.pi*k/L2 for k in range(1, 41)]                            # dyadic wall 15/2 + 2 pi i k/log 2
STRIP = {'f': (-1.0, 0.0), 'D': (0.0, 2.0)}
PREF = {'f': 1.0, 'D': -2.0}


def R(q):
    """analytic factor ZR(q)/DF(-1-q)  (mpmath complex)"""
    q = mm.mpc(q)
    return -(64*mm.sqrt(2)*mm.pi/mm.sin(mm.pi*q/2)*mm.zeta(mm.mpf(13)/2-q)) / \
           ((K-mm.power(2, q))*(1+q)*(2*q-15)*(2*q-5)*mm.zeta(mm.mpf(15)/2-q))


def logZ(q, kind):
    """log(PREF*ZR(q)) up to 2 pi i (only exponentiated differences are used)"""
    return mm.log(PREF[kind]*R(q)) + mm.log(complex(DF(-1-complex(q))))


def S1(q):
    """S'(q) = d/dq log ZR(q)  (complex, exact incl. DF)"""
    q = complex(q); m = mm.mpc(q)
    a, b = mm.mpf(13)/2-m, mm.mpf(15)/2-m
    tq = mm.power(2, m)
    v = (-mm.pi/2*mm.cot(mm.pi*m/2) - mm.zeta(a, 1, 1)/mm.zeta(a) + mm.zeta(b, 1, 1)/mm.zeta(b)
         + tq*L2/(K-tq) - 1/(1+m) - 2/(2*m-15) - 2/(2*m-5))
    p = -1-q
    return complex(v) - complex(DF(p, 1)/DF(p, 0))


def S2(q):
    """S''(q)"""
    q = complex(q); m = mm.mpc(q)
    def lz2(s): z = mm.zeta(s); z1 = mm.zeta(s, 1, 1); z2 = mm.zeta(s, 1, 2); return z2/z-(z1/z)**2
    tq = mm.power(2, m)
    v = ((mm.pi/2)**2/mm.sin(mm.pi*m/2)**2 + lz2(mm.mpf(13)/2-m) - lz2(mm.mpf(15)/2-m)
         + tq*L2**2*K/(K-tq)**2 + 1/(1+m)**2 + 4/(2*m-15)**2 + 4/(2*m-5)**2)
    p = -1-q; d0, d1, d2 = DF(p, 0), DF(p, 1), DF(p, 2)
    return complex(v) + complex(d2/d0-(d1/d0)**2)


def S3(q, h=1e-4):
    return (S2(q+h)-S2(q-h))/(2*h)


def saddle(xi, kind):
    a, b = STRIP[kind]
    return brentq(lambda x: S1(x).real+xi, a+1e-12, b-1e-12, xtol=1e-15, rtol=1e-15, maxiter=200)


def residue_at(qn, xi, kind, which):
    """Res_{q=qn} r^q PREF ZR(q) for a Riemann (7+i gamma) or dyadic (15/2+2 pi i k/log2) pole"""
    m = mm.mpc(qn)
    DFv = complex(DF(-1-complex(qn)))
    num = -(64*mm.sqrt(2)*mm.pi/mm.sin(mm.pi*m/2)*mm.zeta(mm.mpf(13)/2-m))
    if which == 'R':     # 1/Zeta[15/2-q]:  Zeta(15/2-q) ~ -Zeta'(1/2-i gamma)(q-qn)
        den = (K-mm.power(2, m))*(1+m)*(2*m-15)*(2*m-5)*(-mm.zeta(mm.mpf(15)/2-m, 1, 1))
    else:                # 1/(K-2^q):  K-2^q ~ -K log2 (q-qn)
        den = (-K*L2)*(1+m)*(2*m-15)*(2*m-5)*mm.zeta(mm.mpf(15)/2-m)
    return complex(PREF[kind]*mm.exp(xi*m)*num/den)*DFv


class Thimble:
    def __init__(self, xi, kind='D', nGH=80, eps=1e-3, tmax=None, rtol=1e-11, atol=1e-13):
        self.xi, self.kind = xi, kind
        self.q0 = q0 = saddle(xi, kind)
        self.W2 = S2(q0).real
        self.v0 = 1j*np.sqrt(2/self.W2)
        c2 = S3(q0).real/(3*self.W2**2)            # q = q0 + v0 t + c2 t^2 + ...
        self.c2 = c2
        x, w = hermgauss(nGH)
        self.tn, self.wn = x[x > 1e-14], w[x > 1e-14]
        self.w0 = w[np.abs(x) <= 1e-14].sum()
        tmax = tmax or self.tn.max()+0.25
        rhs = lambda t, y: [-2*t/(S1(y[0])+xi)]
        y0 = q0 + self.v0*eps + c2*eps**2
        ev = lambda t, y: 400-abs(y[0]); ev.terminal = True
        self.sol = solve_ivp(rhs, (eps, tmax), [complex(y0)], method='DOP853', dense_output=True,
                             rtol=rtol, atol=atol, events=ev)
        self.eps, self.tend = eps, self.sol.t[-1]
        self.path = self.sol.y[0]

    def q(self, t):
        t = np.atleast_1d(t)
        out = np.empty(len(t), complex)
        small = t < self.eps
        out[small] = self.q0 + self.v0*t[small] + self.c2*t[small]**2
        if (~small).any():
            out[~small] = self.sol.sol(t[~small])[0]
        return out

    def evaluate(self, drift=False):
        """(I, dI/dxi, diagnostics) from the thimble alone"""
        xi, kind = self.xi, self.kind
        ok = self.tn <= self.tend
        t = self.tn[ok]; w = self.wn[ok]
        qs = self.q(t)
        W0 = float(mm.re(logZ(self.q0, kind))) + xi*self.q0
        g = np.empty(len(t), complex); dW = np.empty(len(t), complex)
        for i, (ti, qi) in enumerate(zip(t, qs)):
            dq = -2*ti/(S1(qi)+xi)
            dW[i] = complex(logZ(qi, kind)) + xi*qi - W0 + ti**2       # 0 (mod 2 pi i) on an exact thimble
            g[i] = np.exp(dW[i])*dq
        I = np.exp(W0)/np.pi*((w*g.imag).sum() + self.w0*self.v0.imag/2)
        I1 = np.exp(W0)/np.pi*((w*(qs*g).imag).sum() + self.w0*self.q0*self.v0.imag/2)
        # drift diagnostic: weighted deviation of the path from an exact thimble, relative to the sum
        dq0 = -2*t/np.array([S1(qi)+xi for qi in qs])
        drift = w*np.abs(g-dq0)/max((w*np.abs(dq0)).sum(), 1e-300)
        diag = {'W0': W0, 'q0': self.q0, 'S2': self.W2, 'tend': self.tend, 'nodes_missing': int((~ok).sum()),
                'maxdrift': float(drift.max()) if len(drift) else 0.0}
        return I, I1, diag

    # ---------------- Stokes: trapped poles, zero-terminated paths ----------------
    def _dense_path(self, n=4000):
        t = np.linspace(self.eps, self.tend, n)
        return np.concatenate([[self.q0], self.q(t)])

    def trapped(self, heights, wall, continue_end=True):
        """indices of poles wall+i*h (h in heights) enclosed between Re q=c and the upper thimble:
        odd number of crossings of the rightward ray {Im=h, Re>wall}; the unfinished path is
        continued from its end with its final asymptotic direction."""
        P = self._dense_path()
        x, y = P.real, P.imag
        d = P[-1]-P[-200]
        out = []
        for n, h in enumerate(heights):
            s = (y[:-1]-h)*(y[1:]-h)
            idx = np.where(s <= 0)[0]
            cnt = 0
            for k in idx:
                if y[k+1] == y[k]: continue
                xc = x[k]+(h-y[k])*(x[k+1]-x[k])/(y[k+1]-y[k])
                cnt += xc > wall
            if continue_end and h > y[-1] and d.imag > 0:   # continuation to infinity
                xc = x[-1]+(h-y[-1])*d.real/d.imag
                cnt += xc > wall
            if cnt % 2: out.append(n)
        return out

    def terminal_zero(self, tol=1e-3):
        """if the path ends on a zero 6+i gamma_n of ZR (regime B), return n (0-based)"""
        e = self.path[-1]
        for n, g in enumerate(GAMMA):
            if abs(e-(6+1j*g)) < tol:
                return n
        return None

    def stokes(self):
        """Stokes corrections dI, dI1 (to add to the thimble value) and bookkeeping"""
        xi, kind = self.xi, self.kind
        z = self.terminal_zero()
        # a zero-terminated thimble is continued straight up Re q = 6 (left of both walls): no extra crossings
        iR = self.trapped(GAMMA, 7.0, z is None); iD = self.trapped(DYAD, 7.5, z is None)
        dI = 0j; dI1 = 0j
        for n in iR:
            qn = 7+1j*GAMMA[n]; r = residue_at(qn, xi, kind, 'R'); dI += r; dI1 += qn*r
        for k in iD:
            qk = 7.5+1j*DYAD[k]; r = residue_at(qk, xi, kind, 'D'); dI += r; dI1 += qk*r
        dI, dI1 = -2*dI.real, -2*dI1.real
        tail = tail1 = 0.0
        if z is not None:                           # continue from the zero up Re q = 6 to +i inf
            g0 = GAMMA[z]
            F = lambda y, m: complex(mm.exp(logZ(6+1j*y, kind)+xi*(6+1j*y)))*(6+1j*y)**m
            tail = float(mm.quad(lambda y: F(y, 0).real, [g0, g0+5, g0+20, g0+60]))/np.pi   # |integrand| ~ e^{-pi y/2}
            tail1 = float(mm.quad(lambda y: F(y, 1).real, [g0, g0+5, g0+20, g0+60]))/np.pi
        return dI+tail, dI1+tail1, {'nR': len(iR), 'nD': len(iD), 'zero': z, 'tail': tail,
                                     'resR': dI if iR else 0.0}


def compute(xi, kind='D', nGH=80):
    th = Thimble(xi, kind, nGH)
    I, I1, diag = th.evaluate()
    dI, dI1, st = th.stokes()
    diag.update(st)
    return {'xi': xi, 'I_thimble': I, 'I1_thimble': I1, 'dI': dI, 'dI1': dI1,
            'I': I+dI, 'alpha': (I1+dI1)/(I+dI), 'alpha_thimble': I1/I,
            'dalpha_stokes': (dI1 - (I1/I)*dI)/(I+dI), 'path_end': [th.path[-1].real, th.path[-1].imag], **diag}
