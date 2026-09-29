"""Is the large-r oscillation of xi2 in CorrelationOscillation.nb a k-cutoff artifact?"""
import numpy as np, mpmath as mm
from dfcore import DFfast
from common import cache, PY
import os
DF = DFfast(nGL=200)
H = np.loadtxt(os.path.join(PY, 'htab_CorrelationOscillation.csv'), delimiter=',')   # {kappa, htab} from CorrelationOscillation.nb
lk, lh = np.log(H[:, 0]), np.log(H[:, 1])
Hint = lambda k: np.exp(np.interp(np.log(k), lk, lh))
# 1) is htab our H(kappa)?  H(kappa) = (1/2 pi i) Int kappa^p M(p) dp on Re p = -1.5
def M(p):
    p = mm.mpc(p)
    return complex(mm.gamma(-p)*mm.zeta(p+7.5)/(mm.zeta(p+8.5)*(2*p+7)*(2*p+17)*(1-mm.power(2, -(p+8.5)))))*complex(DF(complex(p)))
y = np.linspace(0, 60, 6001); pc = -1.5+1j*y
Mv = np.array([M(p) for p in pc])
def Hmellin(k):
    g = np.exp(np.log(k)*pc)*Mv; h = y[1]-y[0]
    return (h/3*(g[0]+g[-1]+4*g[1:-1:2].sum()+2*g[2:-1:2].sum())).real/np.pi
for k in [0.1, 1, 10, 100, 1000]:
    print('kappa=%7.1f  htab=%.6e  Mellin H=%.6e  ratio=%.5f' % (k, Hint(k), Hmellin(k), Hint(k)/Hmellin(k)))
# 2) old-style xi2 with cutoffs [0.1, 1000] vs exact alpha_D
k = np.exp(np.linspace(np.log(0.1), np.log(1000), 2_000_001)); w = np.gradient(k); Hk = Hint(k)
def xi2_cut(r, kmin=0.1):
    m = k >= kmin
    x = k[m]*r
    Dv = ((1-np.sin(x)/x)*Hk[m]*w[m]).sum(); DDv = ((-np.cos(x)+np.sin(x)/x)*Hk[m]*w[m]).sum()
    return DDv/Dv
d = np.load(cache('direct_D.npy'))   # xi (natural log), D, alpha_D  exact (no cutoffs)
print('\n log10 r   xi2 with k in[0.1,1000]   exact alpha_D (Mellin)   xi2 with k in [0.01,1000] (htab extrapolated flat)')
for L in [-2, -1, 0, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]:
    r = 10**L; xi = L*np.log(10)
    ex = np.interp(xi, d[0], d[2]) if xi <= 8 else float('nan')
    print('%6.2f   % .6f            % .6f' % (L, xi2_cut(r), ex))
rs = 10**np.linspace(1, 2.5, 301)
np.save(cache('xi2_cut.npy'), np.vstack([np.log10(rs), [xi2_cut(r) for r in rs]]))
