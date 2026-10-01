"""Reference H(xi), dH/dxi by direct integration along Re p = -1.5 (Simpson, mpmath M(p)); cached as cache/lineH_m15.npy.
Used to check the thimble + Stokes representation of the spectrum (thimble_H.py) to ~1e-12 for xi <~ 8."""
import numpy as np, mpmath as mm, os
from dfcore import DFfast
from common import cache
from multiprocessing import Pool
DF = DFfast(nGL=200)
Y = np.linspace(0, 70, 28001)
def M(y):
    mm.mp.dps = 25; p = mm.mpc(-1.5, y)
    return complex(mm.gamma(-p)*mm.zeta(p+7.5)/((2*p+7)*(2*p+17)*mm.zeta(p+8.5)*(1-mm.power(2, -(p+8.5)))))*complex(DF(complex(p)))
def table():
    f = cache('lineH_m15.npy')
    if not os.path.exists(f):
        with Pool(4) as pool: Mv = np.array(pool.map(M, Y, chunksize=200))
        np.save(f, np.vstack([Y, Mv]))
    d = np.load(f); return d[0].real, d[1]
def H_line(xi, Y=None, Mv=None):
    if Y is None: Y, Mv = table()
    h = Y[1]-Y[0]; pc = -1.5+1j*Y
    w = np.full(len(Y), 2.0); w[1:-1:2] = 4.0; w[0] = w[-1] = 1.0; w *= h/3/np.pi
    E = np.exp(np.outer(np.atleast_1d(xi), pc))*Mv
    return (E@w).real, ((E*pc)@w).real
if __name__ == '__main__':
    Y, Mv = table()
    for xi in (4.0, 5.4, 6.0, 7.0, 8.0):
        a = H_line(xi, Y, Mv); b = H_line(xi, Y[::2], Mv[::2])            # step h and 2h
        print('xi=%.1f  H=%.15e  rel. change with step 2h: %.1e' % (xi, a[0][0], abs(b[0][0]/a[0][0]-1)))
