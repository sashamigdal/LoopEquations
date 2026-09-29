import sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_st = open('make_figures.py').read().split("R = json.load")[0]
exec('\n'.join(l for l in _st.splitlines() if not l.startswith(('from thimble', 'OUT =', 'os.makedirs'))))
OUT = sys.argv[1]
S = np.load('spectrum_osc.npy'); O = np.load('oscill.npy')
edge = 2.9153
xs, dnR, dnD = S[0], S[3], S[4]; m = xs >= edge+1.0
lr, daR, daD = O[0], O[5], O[6]; k = lr <= -edge-0.5
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.semilogy(xs[m]-edge, np.abs(dnD[m]), color=C2, lw=1.6, label='energy spectrum, dyadic wall  (δ of d log H/d log κ)')
ax.semilogy(xs[m]-edge, np.abs(dnR[m]), color=C7, lw=1.6, label='energy spectrum, Riemann wall')
ax.semilogy(-edge-lr[k], np.abs(daD[k]), color=C2, lw=1.6, ls='--', label='structure function, dyadic wall  (δ of d log D/d log ρ)')
ax.semilogy(-edge-lr[k], np.abs(daR[k]), color=C7, lw=1.6, ls='--', label='structure function, Riemann wall')
ax.set_xlim(0.5, 5.5); ax.set_ylim(1e-30, 10)
ax.set_xlabel('distance Δ from the convergence edge:  Δ = log κ + log C_min  (spectrum),  Δ = −log ρ + log C_min  (structure function)', fontsize=8.5)
ax.set_ylabel('log-periodic part of the local index'); ax.legend(fontsize=8.3, loc='lower left')
fig.savefig(f'{OUT}/osc_compare.png', dpi=220); plt.close(fig); print('ok')
