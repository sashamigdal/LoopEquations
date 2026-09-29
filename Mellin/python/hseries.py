"""H(k) = (1/2 pi i) Int k^p M(p) dp  (old p-problem, strip -7/2<Re p<0).
Closing right picks up only the Gamma(-p) poles p = n = 0,1,2,...  (zeta(p+15/2)/zeta(p+17/2), 1/(2p+7)(2p+17), odd factor are regular there),
so H(k) = sum_n (-1)^n k^n/n! A_n,  A_n = DF(n) zeta(n+15/2)/(zeta(n+17/2)(2n+7)(2n+17)(1-2^-(n+17/2)))  -- an entire function of k."""
import numpy as np, mpmath as mm
from dfcore import DFfast
DF = DFfast(nGL=200)
mm.mp.dps = 30
def A(n):
    return mm.mpf(complex(DF(float(n))).real)*mm.zeta(n+7.5)/(mm.zeta(n+8.5)*(2*n+7)*(2*n+17)*(1-mm.power(2, -(n+8.5))))
NMAX = 400
An = [A(n) for n in range(NMAX)]
coef = np.array([float((-1)**n*An[n]/mm.factorial(n)) for n in range(NMAX)])
def Hser(k):
    k = np.asarray(k, float)
    # Horner in double is fine for k <= ~40 (terms ~ (C k)^n/n!, C ~ 0.06)
    out = np.zeros_like(k)
    for c in coef[::-1]: out = out*k + c
    return out
if __name__ == '__main__':
    def M(p):
        p = mm.mpc(p)
        return complex(mm.gamma(-p)*mm.zeta(p+7.5)/(mm.zeta(p+8.5)*(2*p+7)*(2*p+17)*(1-mm.power(2, -(p+8.5)))))*complex(DF(complex(p)))
    y = np.linspace(0, 60, 6001); pc = -1.5+1j*y
    Mv = np.array([M(p) for p in pc]); h = y[1]-y[0]
    def Hline(k):
        g = np.exp(np.log(k)*pc)*Mv
        return (h/3*(g[0]+g[-1]+4*g[1:-1:2].sum()+2*g[2:-1:2].sum())).real/np.pi
    print('H(0) =', coef[0], ' (pi/2)H(0) =', np.pi/2*coef[0], ' vs large-r coefficient of f: 0.05835907136')
    for k in [0.1, 0.5, 1, 3, 10, 20, 40]:
        print('k=%5.1f  series=%.12e  line=%.12e  rel=%.1e' % (k, Hser(k), Hline(k), abs(Hser(k)/Hline(k)-1)))
    np.save('hcoef.npy', coef)
