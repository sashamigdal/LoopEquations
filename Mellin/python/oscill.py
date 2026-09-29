"""Full log-periodic (wall-pole) part of D(r) and alpha_D on a fine grid, plus exact D, alpha_D there."""
import numpy as np
from thimble import residue_at, GAMMA, DYAD
line = np.load('lineD_c1.npy'); y = line[0].real; Z = line[1]; q = 1.0 + 1j*y; h = y[1]-y[0]
simp = lambda g: h/3*(g[0]+g[-1]+4*g[1:-1:2].sum()+2*g[2:-1:2].sum())
xis = np.round(np.arange(-8, 8.0001, 0.01), 6)
D = np.array([simp(np.exp(x*q)*Z).real/np.pi for x in xis]); D1 = np.array([simp(q*np.exp(x*q)*Z).real/np.pi for x in xis])
alpha = D1/D
poles = [(7+1j*g, 'R') for g in GAMMA] + [(7.5+1j*d, 'D') for d in DYAD]
c = np.array([residue_at(qn, 0.0, 'D', w) for qn, w in poles]); qn = np.array([p for p, _ in poles])
E = np.exp(np.outer(xis, qn))
W = -2*(E*c).sum(1).real; W1 = -2*(E*c*qn).sum(1).real
dalpha_wall = (W1 - alpha*W)/D
# separate Riemann and dyadic parts
nR = len(GAMMA)
WR = -2*(E[:, :nR]*c[:nR]).sum(1).real; WR1 = -2*(E[:, :nR]*c[:nR]*qn[:nR]).sum(1).real
WD = W-WR; WD1 = W1-WR1
np.save('oscill.npy', np.vstack([xis, D, alpha, W, dalpha_wall, (WR1-alpha*WR)/D, (WD1-alpha*WD)/D]))
for x in [-8, -7, -6, -5, -4.5, -4, -3.5, -3]:
    i = int(round((x+8)/0.01)); print('xi=%5.2f D=%.6e alpha=%.8f  wall part of D (rel) = % .2e  dalpha_wall = % .2e (Riemann % .2e, dyadic % .2e)' % (x, D[i], alpha[i], W[i]/D[i], dalpha_wall[i], (WR1-alpha*WR)[i]/D[i], (WD1-alpha*WD)[i]/D[i]))
