"""(1) check Z(q) = -Gamma(-1-q) cos(pi q/2) M_odd(-1-q) == ZR(q);
(2) energy spectrum H(xi) (odd ensemble) and its log-periodic (wall-pole) part vs the structure-function one."""
import numpy as np, mpmath as mm
from dfcore import DFfast
from thimble import R as Ran, GAMMA, DYAD
from common import cache
DF = DFfast(nGL=200); mm.mp.dps = 25
def M(p, eta=1):
    p = mm.mpc(p)
    return mm.gamma(-p)*mm.zeta(p+7.5)/((2*p+7)*(2*p+17)*mm.zeta(p+8.5)*(1-eta*mm.power(2, -(p+8.5))))*complex(DF(complex(p)))
for q in [mm.mpc(-0.4, 0), mm.mpc(0.7, 3.2), mm.mpc(5.1, -7.3)]:
    lhs = -mm.gamma(-1-q)*mm.cos(mm.pi*q/2)*M(-1-q)
    rhs = Ran(q)*complex(DF(-1-complex(q)))
    print('Z check q=%s  ratio=%s' % (mm.nstr(q, 3), mm.nstr(lhs/rhs, 15)))
# H(xi) on a line Re p = -1.5, and dH/dxi
y = np.linspace(0, 70, 14001); pc = -1.5+1j*y; h = y[1]-y[0]
Mv = np.array([complex(M(p)) for p in pc])
wS = np.full(len(y), 2.0); wS[1:-1:2] = 4.0; wS[0] = wS[-1] = 1.0; wS *= h/3/np.pi
xs = np.round(np.arange(0.0, 12.0001, 0.01), 6)
E = np.exp(np.outer(xs, pc))*Mv
H = (E@wS).real; H1 = ((E*pc)@wS).real; n = H1/H          # n(xi) = d log H / d xi  (effective spectral index)
# wall residues of kappa^p M(p): Riemann p = -8 + i rho (from 1/zeta(p+17/2)), dyadic p = -17/2 + 2 pi i m/log2
def res_R(rho):
    p = mm.mpc(-8, rho); s = p+mm.mpf(17)/2
    return complex(mm.gamma(-p)*mm.zeta(p+7.5)/((2*p+7)*(2*p+17)*mm.zeta(s, 1, 1)*(1-mm.power(2, -s))))*complex(DF(complex(p))), complex(p)
def res_D(m):
    p = mm.mpc(-8.5, 2*mm.pi*m/mm.log(2))
    return complex(mm.gamma(-p)*mm.zeta(p+7.5)/((2*p+7)*(2*p+17)*mm.zeta(p+8.5)*mm.log(2)))*complex(DF(complex(p))), complex(p)
RR = [res_R(g) for g in GAMMA]; RD = [res_D(m) for m in range(1, len(DYAD)+1)]
def wall(parts):
    c = np.array([a for a, _ in parts]); p = np.array([b for _, b in parts]); Ex = np.exp(np.outer(xs, p))
    return 2*(Ex*c).sum(1).real, 2*(Ex*c*p).sum(1).real
WR, WR1 = wall(RR); WD, WD1 = wall(RD)
dn_R = (WR1-n*WR)/H; dn_D = (WD1-n*WD)/H
np.save(cache('spectrum_osc.npy'), np.vstack([xs, H, n, dn_R, dn_D, WR/H, WD/H]))
Cmax = float(np.exp(DF.lC.max())); print('left-closed series for H converges for kappa > 1/C_min = %.2f (xi > %.2f)' % (1/np.exp(DF.lC.min()), -DF.lC.min()))
for x in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    i = int(round(x/0.01)); print('xi=%4.1f  H=%.4e  n=%.4f  wall/H: Riemann %.2e dyadic %.2e | dn: Riemann %.2e dyadic %.2e' % (x, H[i], n[i], WR[i]/H[i], WD[i]/H[i], dn_R[i], dn_D[i]))
# why the spectrum oscillates much more: Gamma(-p) at the walls vs csc(pi q/2) in coordinate space (q = -1-p)
pD = mm.mpc(-8.5, 2*mm.pi/mm.log(2)); pR = mm.mpc(-8, GAMMA[0])
print('C(Delta) in [e^%.3f, e^%.3f]' % (DF.lC.min(), DF.lC.max()))
print('|Gamma(-p)| at the first dyadic pole p = -17/2 + 2 pi i/log2: %.3g;  |csc(pi q/2)| at q = 15/2 + 2 pi i/log2: %.3g' % (
    abs(mm.gamma(-pD)), abs(1/mm.sin(mm.pi*(-1-pD)/2))))
print('|Gamma(-p)| at the first Riemann pole p = -8 + i gamma_1: %.3g;  |csc(pi q/2)| at q = 7 + i gamma_1: %.3g' % (
    abs(mm.gamma(-pR)), abs(1/mm.sin(mm.pi*(-1-pR)/2))))
