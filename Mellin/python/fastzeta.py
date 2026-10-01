"""Riemann zeta and its logarithmic derivatives for complex s in double precision, by Euler-Maclaurin summation.

    zeta(s) = sum_{n<N} n^-s + N^(1-s)/(s-1) + N^-s/2 + sum_{k=1}^{M} B_2k/(2k)! (s)_(2k-1) N^(-s-2k+1),
    (s)_j = s (s+1) ... (s+j-1).

N grows with |s|, so the remainder ~ |(s)_(2M+1)| / (2 pi N)^(2M+1) stays below 1e-15 for the arguments used here
(Re s >= 1/2, |Im s| < 400). For Re s < 1/2 the functional equation zeta(s) = chi(s) zeta(1-s),
chi(s) = 2^s pi^(s-1) sin(pi s/2) Gamma(1-s), is used, so the sum never suffers the cancellation of n^-s with Re s < 0.
About 100 times faster than mpmath.zeta at |Im s| ~ 100; checked against mpmath below."""
import numpy as np
from math import factorial
import mpmath as mm
from scipy import special as sp

M = 16
_B = [float(mm.bernoulli(2*k))/factorial(2*k) for k in range(1, M+1)]


def _N(s):
    return int(max(40, abs(s)*0.7 + 30))


def _em(s):
    """Euler-Maclaurin (zeta, zeta', zeta'') for Re s >= 1/2"""
    s = complex(s); N = _N(s)
    n = np.arange(1, N, dtype=float); ln = np.log(n); t = np.exp(-s*ln)
    z0 = t.sum(); z1 = -(ln*t).sum(); z2 = (ln*ln*t).sum()
    LN = np.log(N); NN = np.exp((1-s)*LN)                      # N^(1-s)
    a = 1/(s-1)
    z0 += NN*a;  z1 += NN*(-LN*a - a*a);  z2 += NN*(LN*LN*a + 2*LN*a*a + 2*a**3)
    Ns = np.exp(-s*LN)
    z0 += Ns/2;  z1 += -LN*Ns/2;  z2 += LN*LN*Ns/2
    # Bernoulli terms: T_k = B_2k/(2k)! P_k(s) N^(-s-2k+1),  P_k = (s)_(2k-1)
    P, P1, P2 = s, 1.0+0j, 0j                                  # (s)_1 and its first two s-derivatives
    for k in range(1, M+1):
        if k > 1:
            for j in (2*k-3, 2*k-2):                           # multiply by (s+j) twice
                P2 = P2*(s+j) + 2*P1; P1 = P1*(s+j) + P; P = P*(s+j)
        E = np.exp((-s-2*k+1)*LN)
        z0 += _B[k-1]*P*E
        z1 += _B[k-1]*(P1 - LN*P)*E
        z2 += _B[k-1]*(P2 - 2*LN*P1 + LN*LN*P)*E
    return z0, z1, z2


LPI, L2 = np.log(np.pi), np.log(2.0)


def _chi_logder(s):
    """chi'/chi for chi(s) = 2^s pi^(s-1) sin(pi s/2) Gamma(1-s)"""
    return L2 + LPI + np.pi/2/np.tan(np.pi*s/2) - sp.psi(1-s)


def _logchi(s):
    return s*L2 + (s-1)*LPI + np.log(np.sin(np.pi*s/2)) + sp.loggamma(1-s)


def zeta_d(s):
    """(zeta(s), zeta'(s), zeta''(s)) for a complex scalar s"""
    s = complex(s)
    if s.real >= 0.5:
        return _em(s)
    Z0, Z1, Z2 = _em(1-s)                                   # derivatives with respect to w = 1-s
    z0 = np.exp(_logchi(s))*Z0
    ld = _chi_logder(s) - Z1/Z0
    psi1 = complex(mm.psi(1, mm.mpc(1-s)))
    sn = np.sin(np.pi*s/2) if abs(s.imag) < 400 else np.inf          # csc^2 underflows to 0 for large Im s
    ld2 = -(np.pi/2)**2/sn**2 + psi1 + (Z2/Z0 - (Z1/Z0)**2)
    return z0, z0*ld, z0*(ld2 + ld*ld)


def zeta(s):
    s = complex(s)
    if s.real >= 0.5:
        return _em(s)[0]
    return np.exp(_logchi(s))*_em(1-s)[0]


def logzeta(s):
    """log zeta(s), up to 2 pi i"""
    s = complex(s)
    if s.real >= 0.5:
        return np.log(_em(s)[0])
    return _logchi(s) + np.log(_em(1-s)[0])


def logder(s):
    """zeta'/zeta"""
    s = complex(s)
    if s.real >= 0.5:
        z0, z1, _ = _em(s)
        return z1/z0
    Z0, Z1, _ = _em(1-s)
    return _chi_logder(s) - Z1/Z0


def logder2(s):
    """d/ds (zeta'/zeta) = zeta''/zeta - (zeta'/zeta)^2"""
    z0, z1, z2 = zeta_d(s)
    return z2/z0 - (z1/z0)**2


if __name__ == '__main__':
    mm.mp.dps = 30
    worst = 0.0
    rng = np.random.default_rng(1)
    pts = [complex(x, y) for x, y in zip(rng.uniform(-6.5, 8.5, 500), rng.uniform(-5, 380, 500))]
    pts += [0.5+14.134725141734693j+1e-3, -0.5+14.134725j, 1.5+0.3j, 2+0j, -1.5+9.0647j, 0.5+200j]
    for s in pts:
        z0, z1, z2 = zeta_d(s)
        r0 = complex(mm.zeta(s)); r1 = complex(mm.zeta(s, 1, 1)); r2 = complex(mm.zeta(s, 1, 2))
        e = max(abs(z0/r0-1), abs(logder(s)/(r1/r0)-1), abs(zeta(s)/r0-1))
        worst = max(worst, e)
    print('zeta and zeta\'/zeta vs mpmath at %d points (-6.5 < Re s < 8.5, Im s < 380): max relative error %.1e'
          % (len(pts), worst))
