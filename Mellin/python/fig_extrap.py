"""README figure 12: Re -> infinity extrapolations in 1/Re and Re^-1/2 (needs the MPI data)."""
import json, sys, numpy as np
from common import *
plt = style()
from finite_box import alphaL
OUT = sys.argv[1] if len(sys.argv) > 1 else RESULTS
J = load_json(cache('mpi_extrapolate_Re_full.json')); runs = J['runs']
fig, axs = plt.subplots(1, 2, figsize=(13, 6.0), sharey=True, gridspec_kw={'wspace': 0.06})
for ax, key, title in [(axs[0], 'mid+high (Re>=1046)|1.0', 'extrapolated from Re_λ = 1046 … 5779 in 1/Re_λ'),
                       (axs[1], 'mid+high (Re>=1046)|0.5', 'extrapolated from Re_λ = 1046 … 5779 in Re_λ^{-1/2}')]:
    E = J['extrap'][key]; x = np.array(E['x']); ai = np.array(E['ainf']); ae = np.array(E['aerr'])
    k = 0
    for r in runs:
        if r['Re'] < 1000: continue
        xx = np.array(r['lr'])-np.log(r['L_eta']); ax.plot(xx, r['a'], '.', ms=2.5, color=SEQ[3+k], alpha=0.8, label='Re_λ = %d' % round(r['Re'])); k += 1
    ok = np.isfinite(ai)
    ax.errorbar(x[ok], ai[ok], yerr=np.where(np.isfinite(ae[ok]), ae[ok], 0), fmt='o', ms=5, color=INK, mfc=SURF, mew=1.3, elinewidth=1, capsize=0, label='Re_λ → ∞ extrapolation')
    xs = np.linspace(-4, 2.7, 700)
    ax.plot(xs, alphaL(xs-E['s'], 0.0)[0], color=C1, lw=2, label='theory, infinite system (s = %.2f, χ²/dof %.1f)' % (E['s'], E['chi2']))
    ax.plot(xs, alphaL(xs-E['s2'], E['kmin'])[0], color=C2, lw=1.6, ls='--', label='theory + box k_min = %.1f (χ²/dof %.1f)' % (E['kmin'], E['chi2_2']))
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xlim(-4, 2.7); ax.set_ylim(-0.15, 0.8)
    ax.set_xlabel('log(r / L),  L = integral scale from each run'); ax.set_title(title, fontsize=10.5, loc='left')
axs[0].set_ylabel('α = d log S₂ / d log r')
for ax in axs: ax.legend(fontsize=7.5, loc='lower left', ncol=2)
fig.suptitle('Max Planck index at mid/high Re, extrapolated to Re_λ → ∞, vs the infinite-Re theory', x=0.01, ha='left', fontsize=11.5)
fig.savefig(f'{OUT}/fig12_extrapolated_Re_inf.png'); plt.close(fig)
print('ok')
