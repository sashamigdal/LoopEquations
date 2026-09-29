"""README figure 13 (with titles; the paper version is made by paper_figures.py; needs the MPI data)."""
import json, sys, numpy as np
from common import *
plt = style()
from finite_box import alphaL
OUT = sys.argv[1] if len(sys.argv) > 1 else RESULTS
J = load_json(cache('mpi_extrap_invRe_full.json')); x = np.array(J['x']); Re = np.array(J['Re']); B = np.array([[np.nan if v is None else v for v in row] for row in J['binned']])
F = J['fits']; A = F['Re>=1046 (adopted)']
ai, ae = np.array(A['ainf'], float), np.array(A['aerr'], float); s = A['s']
fig = plt.figure(figsize=(13, 7.6)); gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1], width_ratios=[1.35, 1], hspace=0.12, wspace=0.22)
ax = fig.add_subplot(gs[0, 0]); axr = fig.add_subplot(gs[1, 0], sharex=ax); axl = fig.add_subplot(gs[:, 1])
ok = np.isfinite(ai) & np.isfinite(ae)
ax.errorbar(x[ok], ai[ok], yerr=ae[ok], fmt='o', ms=5, color=INK, mfc=SURF, mew=1.3, elinewidth=1, label='data extrapolated to Re_λ → ∞ (in 1/Re_λ, runs Re_λ ≥ 1046)')
xs = np.linspace(-4, 2.7, 700)
ax.plot(xs, alphaL(xs-s, 0.0)[0], color=C1, lw=2.2, label='theory α_D(ρ), ρ = r/ℓ_D, ℓ_D/L = e^s = %.2f ± %.2f' % (np.exp(s), np.exp(s)*A['ds']))
for name, c in [('all 11 runs (Re>=413)', C3), ('Re>=2398', C2)]:
    ax.plot(xs, alphaL(xs-F[name]['s'], 0.0)[0], color=c, lw=1.2, ls='--', label='theory, s fitted to the %s extrapolation (s = %.2f)' % ('all-runs' if 'all' in name else 'Re_λ ≥ 2398', F[name]['s']))
ax.axhline(0, color=INK2, lw=0.8); ax.set_xlim(-4, 2.7); ax.set_ylim(-0.15, 0.8); ax.set_ylabel('α_∞ = d log S₂/d log r')
x_fit0 = x[(ai < 0.355) & (x > -3.0) & np.isfinite(ai)].min()
ax.axvspan(-4, x_fit0-0.05, color=GRID, alpha=0.5, lw=0); ax.text(-3.95, 0.02, 'not fitted: α_∞ > 0.355\n(inertial range and below)', fontsize=8.5, color=INK2)
ax.legend(fontsize=8.3, loc='upper right'); plt.setp(ax.get_xticklabels(), visible=False)
ax.set_title('Re_λ → ∞ index vs theory (χ²/dof %.1f, weighted rms %.3f)' % (A['chi2dof'], A['wrms']), fontsize=10.5, loc='left')
fit = (ai < 0.355) & (x > -3.0) & ok
res = ai-alphaL(x-s, 0.0)[0]
axr.errorbar(x[fit], res[fit], yerr=ae[fit], fmt='o', ms=4, color=INK, mfc=SURF, mew=1.1, elinewidth=1)
axr.axhline(0, color=C1, lw=1.5); axr.set_ylim(-0.12, 0.12); axr.set_xlabel('log(r / L)'); axr.set_ylabel('α_∞ − α_D')
# linearity in 1/Re at selected x
sel = Re >= 1000; X = 1/Re
for x0, c in zip([-1.0, 0.0, 0.5, 1.0, 1.5], [C7, C1, C3, C4, C2]):
    j = int(np.argmin(np.abs(x-x0))); y = B[sel, j]; o = np.isfinite(y)
    axl.plot(1e3*X[sel][o], y[o], 'o', color=c, ms=6, mec=SURF, mew=1.2)
    xx = np.linspace(0, 1e3*X[sel].max()*1.05, 20)
    axl.plot(xx, ai[j]+A['slope'][j]*xx/1e3, color=c, lw=1.4, label='log(r/L) = %.1f' % x0)
    axl.errorbar([0], [ai[j]], yerr=[ae[j]], fmt='s', color=c, ms=6, mfc=SURF, mew=1.5)
axl.set_xlim(-0.03, 1e3*X[sel].max()*1.08); axl.set_xlabel('10³ / Re_λ'); axl.set_ylabel('α at fixed r/L')
axl.legend(fontsize=8.5, loc='upper left'); axl.set_title('α vs 1/Re_λ at fixed r/L (squares: Re_λ → ∞)', fontsize=10.5, loc='left')
fig.savefig(f'{OUT}/fig13_invRe_extrapolation.png'); plt.close(fig); print('ok')
