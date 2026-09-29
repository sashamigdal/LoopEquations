import json, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from finite_box import alphaL
_st = open('make_figures.py').read().split("R = json.load")[0]
exec('\n'.join(l for l in _st.splitlines() if not l.startswith(('from thimble', 'OUT =', 'os.makedirs'))))   # style + palette only
OUT = sys.argv[1]
F = json.load(open('../mpi/fit_box.json')); G = json.load(open('../mpi/fit_box_global.json')); Lam = G['best']['Lambda']
# ---- 9: per-run comparison, small multiples ----
fig, axs = plt.subplots(4, 3, figsize=(11, 12), sharey=True)
for ax, (j, f) in zip(axs.flat, enumerate(F)):
    lr, a, i0 = np.array(f['lr']), np.array(f['a']), f['i0']; rmax = np.exp(lr[-1])
    x = np.linspace(lr[i0]-1.5, lr[-1]+0.4, 500)
    ax.plot(lr, a, 'o', ms=3, mfc='none', mec=INK2, mew=0.7, label='Max Planck')
    ax.plot(x, alphaL(x-f['s_inf'], 0.0)[0], color=INK2, lw=1.4, ls='--', label='infinite system (fit s)')
    ax.plot(x, alphaL(x-f['s_box'], f['kmin'])[0], color=C1, lw=1.8, label='finite box, own k_min')
    sG = G['best']['shifts'][j]
    ax.plot(x, alphaL(x-sG, np.pi*np.exp(sG)/(Lam*rmax))[0], color=C2, lw=1.4, label='finite box, common L = %.2f r_max' % Lam)
    ax.axhline(0, color=GRID, lw=1); ax.axvline(lr[i0], color=GRID, lw=1)
    ax.set_xlim(lr[i0]-1.5, lr[-1]+0.4); ax.set_ylim(-0.15, 0.6)
    ax.set_title('Re_λ = %d   rms %.3f → %.3f' % (round(f['Re']), f['rms_inf'], f['rms_box']), fontsize=9.5, loc='left')
    ax.set_xlabel('log(r/η)', fontsize=9)
axs.flat[-1].axis('off'); h, l = axs.flat[0].get_legend_handles_labels(); axs.flat[-1].legend(h, l, loc='center', fontsize=10)
for ax in axs[:, 0]: ax.set_ylabel('α = d log S₂/d log r')
fig.suptitle('Large-r index: Max Planck data vs infinite-system and finite-box (k > k_min = π/L) theory', x=0.02, ha='left', fontsize=11.5)
fig.tight_layout(); fig.savefig(f'{OUT}/fig9_finite_box_runs.png'); plt.close(fig)
# ---- 10: what the cutoff does to alpha (theory units) ----
fig, ax = plt.subplots(figsize=(8.4, 4.8))
x = np.linspace(-3, 4, 1400)
ax.plot(x, alphaL(x, 0.0)[0], color=INK, lw=2, label='infinite system')
for km, c in zip([3.0, 1.0, 0.3, 0.1], [C1, C3, C2, C7]):
    ax.plot(x, alphaL(x, km)[0], color=c, lw=1.5, label='κ_min = %.1f  (W/L = π/κ_min = %.1f)' % (km, np.pi/km))
ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-0.12, 0.6); ax.set_xlabel('log ρ'); ax.set_ylabel('α_W = d log D_W / d log ρ')
ax.legend(fontsize=9); ax.set_title('Finite box: α oscillates about 0 with period 2π/k_min in r, first dip near r ≈ L', fontsize=10.5, loc='left')
fig.savefig(f'{OUT}/fig10_finite_box_theory.png'); plt.close(fig)
# ---- 11: fitted box size vs Re ----
fig, ax = plt.subplots(figsize=(6.4, 4.2))
Re = np.array([f['Re'] for f in F]); Lf = np.array([f['L_over_eta'] for f in F]); rm = np.array([f['rmax_over_eta'] for f in F])
ok = np.array([f['kmin'] > 1e-3 for f in F])
ax.loglog(Re[ok], Lf[ok], 'o', color=C1, ms=7, mec=SURF, mew=1.5, label='fitted L/η = π e^s / k_min')
ax.loglog(Re, rm, 's', color=INK2, ms=5, mfc='none', label='largest measured r/η')
ax.set_xlabel('Re_λ'); ax.set_ylabel('length / η'); ax.legend(fontsize=9)
ax.set_title('Fitted box size tracks the largest separation', fontsize=10.5, loc='left')
fig.savefig(f'{OUT}/fig11_box_size_vs_Re.png'); plt.close(fig)
print('ok')
