"""Ground truth for D(r) = 2(f(0)-f(r)) = (1/2 pi i) Int_{c'} r^q (-2 ZR(q)) dq, 0<c'<2, and alpha_D = dlogD/dlogr."""
import numpy as np, mpmath as mm, time
from dfcore import DFfast
from multiprocessing import Pool
DF = DFfast(nGL=200)
s2 = np.sqrt(2.0)
def Ran(z):
    z = complex(z)
    return complex(-(64*s2*mm.pi/mm.sin(mm.pi*z/2)*mm.zeta(6.5-z))/((128*s2-2**z)*(1+z)*(2*z-15)*(2*z-5)*mm.zeta(7.5-z)))
c = 1.0
y = np.linspace(0, 70, 28001)
q = c + 1j*y
if __name__ == '__main__':
    t = time.time()
    with Pool(4) as p: R = np.array(p.map(Ran, list(q), chunksize=500))
    Z = -2*R*DF(-1-q)
    np.save('lineD_c1.npy', np.vstack([y, Z])); print('line', time.time()-t)
    h = y[1]-y[0]
    def simpson(g): return h/3*(g[0] + g[-1] + 4*g[1:-1:2].sum() + 2*g[2:-1:2].sum())
    xis = np.arange(-8, 8.001, 0.02)
    D = np.array([simpson(np.exp(x*q)*Z).real/np.pi for x in xis])
    D1 = np.array([simpson(q*np.exp(x*q)*Z).real/np.pi for x in xis])   # dD/dxi
    np.save('direct_D.npy', np.vstack([xis, D, D1/D]))
    for x, d, a in list(zip(xis, D, D1/D))[::25]:
        print('%6.2f  D=% .8e  alphaD=% .8f' % (x, d, a))
