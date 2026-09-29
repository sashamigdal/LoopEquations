"""Checks quoted in the paper and the README:
  1. S'(q), S''(q) against finite differences of log Z(q);
  2. the Riemann and dyadic residue formulas against circle integrals;
  3. the thimble + Gauss-Hermite + Stokes scan (../results/scan_D.json) against the direct contour integral
     (cache/direct_D.npy, from direct_D.py) over log rho in [-8, 8], and the Stokes terms against the full wall sum (cache/oscill.npy);
  4. writes ../results/pythonReference_D.csv (the reference values checked by the Mathematica driver)."""
import json, numpy as np, mpmath as mm
from thimble import logZ, S1, S2, R, DF, PREF, GAMMA, DYAD, residue_at
from common import cache, res, load_json

for q in [0.7, -0.4, 1.3+2.1j, 5.2+8j]:
    h = 1e-5
    n1 = (complex(logZ(q+h, 'D'))-complex(logZ(q-h, 'D')))/(2*h); n2 = (S1(q+h)-S1(q-h))/(2*h)
    print("q=%-10s |S' - FD| = %.1e   |S'' - FD| = %.1e" % (q, abs(n1-S1(q)), abs(n2-S2(q))))
for qn, which in [(7+1j*GAMMA[0], 'R'), (7.5+1j*DYAD[0], 'D')]:
    xi, e = -5.0, 1e-3
    f = lambda t: (mm.exp(xi*(qn+e*mm.exp(1j*t)))*PREF['D']*R(qn+e*mm.exp(1j*t))*complex(DF(-1-complex(qn+e*mm.exp(1j*t))))
                   * e*1j*mm.exp(1j*t))
    circ = complex(mm.quad(f, [0, mm.pi, 2*mm.pi]))/(2j*np.pi); rr = residue_at(qn, xi, 'D', which)
    print('%s residue at %s: formula vs circle integral, rel. diff %.1e' % ('Riemann' if which == 'R' else 'dyadic', np.round(qn, 4), abs(rr/circ-1)))

S = load_json(res('scan_D.json')); d = np.load(cache('direct_D.npy'))
xs = np.array([r['xi'] for r in S]); DD = np.array([r['I'] for r in S]); aa = np.array([r['alpha'] for r in S])
k = np.rint((xs-d[0][0])/(d[0][1]-d[0][0])).astype(int); ok = np.abs(d[0][k]-xs) < 1e-9     # scan points on the 0.02 grid
relD = np.abs(DD[ok]/d[1][k[ok]]-1); da = np.abs(aa[ok]-d[2][k[ok]])
print('scan vs direct contour integral, %d points in [%.1f, %.1f]: max |dD/D| = %.1e, max |d alpha| = %.1e'
      % (ok.sum(), xs[ok].min(), xs[ok].max(), relD.max(), da.max()))
tail = max(abs(r['tail']/r['I']) for r in S if r['zero'] is not None)
print('zero-terminated thimbles: largest tail up Re q = 6, relative to D: %.1e' % tail)
o = np.load(cache('oscill.npy'))                   # full wall sum W(xi) from oscill.py
full = [r for r in S if r['nR'] == len(GAMMA) and r['nD'] == len(DYAD)]
dev = max(abs(r['dI']-np.interp(r['xi'], o[0], o[3]))/r['I'] for r in full)
print('where all %d + %d wall poles are trapped (%d points): Stokes terms vs full wall sum, max |diff|/D = %.1e' % (len(GAMMA), len(DYAD), len(full), dev))
fd = max(r['xi'] for r in S if r['nD'] == len(DYAD))
print('the lowest dyadic pole (height 2 pi/log 2 = %.2f) is trapped for log rho <= %.2f' % (DYAD[0], fd))
print('regimes: zero-terminated for log rho in [%.2f, %.2f]; walls trapped for log rho <= %.2f' % (
    min(r['xi'] for r in S if r['zero'] is not None), max(r['xi'] for r in S if r['zero'] is not None),
    max(r['xi'] for r in S if r['nR']+r['nD'] > 0 and r['zero'] is None)))
with open(res('pythonReference_D.csv'), 'w') as fh:
    fh.write('log r,D,alpha,q0,trapped Riemann,trapped dyadic,terminal zero n (1-based)\n')
    for x0 in [3.0, 0.0, -2.0, -3.0, -3.5, -4.5, -6.0]:
        r = next(r for r in S if abs(r['xi']-x0) < 1e-9)
        fh.write('%.1f,%.12e,%.12f,%.10f,%d,%d,%s\n' % (x0, r['I'], r['alpha'], r['q0'], r['nR'], r['nD'], '' if r['zero'] is None else r['zero']+1))
print('wrote', res('pythonReference_D.csv'))
