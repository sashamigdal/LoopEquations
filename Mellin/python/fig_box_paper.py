"""Per-run figure for the paper: infinite system, per-run k_min, and one physical width W for all runs."""
import json, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from finite_box import alphaL
_st = open('make_figures.py').read().split("R = json.load")[0]
exec('\n'.join(l for l in _st.splitlines() if not l.startswith(('from thimble', 'OUT =', 'os.makedirs'))))
OUT = sys.argv[1]; W = float(sys.argv[2]) if len(sys.argv) > 2 else 2.34
F = json.load(open('../mpi/fit_box.json')); P = json.load(open('../mpi/physical_units.json'))
fig, axs = plt.subplots(4, 3, figsize=(11, 12), sharey=True)
for ax, f, p in zip(axs.flat, F, P):
    lr, a, i0 = np.array(f['lr']), np.array(f['a']), f['i0']; X, A = lr[i0:], a[i0:]; eta = p['eta']
    e = lambda s: np.mean((A-alphaL(X-s, np.pi*eta*np.exp(s)/W)[0])**2)
    sW = minimize_scalar(e, bounds=(f['s_inf']-1.5, f['s_inf']+1.5), method='bounded').x
    x = np.linspace(lr[i0]-1.5, lr[-1]+0.4, 500)
    ax.plot(lr, a, 'o', ms=3, mfc='none', mec=INK2, mew=0.7, label='Max Planck')
    ax.plot(x, alphaL(x-f['s_inf'], 0.0)[0], color=INK2, lw=1.4, ls='--', label='infinite system')
    ax.plot(x, alphaL(x-f['s_box'], f['kmin'])[0], color=C1, lw=1.8, label='finite width, κ_min fitted per run')
    ax.plot(x, alphaL(x-sW, np.pi*eta*np.exp(sW)/W)[0], color=C2, lw=1.4, label='finite width, one W = %.2f m for all runs' % W)
    ax.axhline(0, color=GRID, lw=1); ax.axvline(lr[i0], color=GRID, lw=1)
    ax.set_xlim(lr[i0]-1.5, lr[-1]+0.4); ax.set_ylim(-0.15, 0.6)
    ax.text(0.03, 0.05, 'Re_λ = %d\nrms %.3f → %.3f' % (round(f['Re']), f['rms_inf'], f['rms_box']), transform=ax.transAxes, fontsize=9)
    ax.set_xlabel('log(r/η)', fontsize=9)
axs.flat[-1].axis('off'); h, l = axs.flat[0].get_legend_handles_labels(); axs.flat[-1].legend(h, l, loc='center', fontsize=10)
for ax in axs[:, 0]: ax.set_ylabel('α = d log S₂/d log r')
fig.tight_layout(); fig.savefig(f'{OUT}/fig9_finite_box_runs.png', dpi=200); plt.close(fig); print('ok')
