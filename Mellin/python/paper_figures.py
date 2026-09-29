"""The computed figures of the paper, written to ../paper/figs (or the directory given as the first argument):

  fig1_D_alpha.png                 D(rho) and alpha(log rho): thimbles vs the direct contour integral
  fig3_staircase.png               Stokes staircases of the structure function
  fig4_thimbles.png                thimbles in the q plane
  osc_compare.png                  log-periodic parts of the spectral index and of alpha_D
  fig10_finite_box_theory.png      alpha_W for several kappa_min
  fig9_finite_box_runs.png         Max Planck runs vs infinite system / per-run kappa_min / one width W   (needs the MPI data)
  fig13_invRe_extrapolation.png    1/Re_lambda extrapolation vs theory                                    (needs the MPI data)

Inputs: ../results/scan_D.json (scan.py), cache/oscill.npy (oscill.py), cache/spectrum_osc.npy (spectrum_osc.py),
cache/mpi_*_full.json (finite_box.py, physical_units.py, extrapolate_Re.py, extrap_invRe.py).
RegularPolygons.png, StokesOddStaircase.png and BSSpectra_clean.png are reproduced from Refs. [ReviewPaperAM, migdal2026Riemann]."""
import os, sys, numpy as np
from common import (cache, res, load_json, have_mpi, style, PAPER_FIGS, INK, INK2, GRID, SURF,
                    C1, C2, C3, C4, C5, C6, C7, LOGCMIN)
plt = style(titles=False)
from scipy.optimize import minimize_scalar
from thimble import Thimble, GAMMA, DYAD
from finite_box import alphaL

OUT = sys.argv[1] if len(sys.argv) > 1 else PAPER_FIGS
os.makedirs(OUT, exist_ok=True)
R = load_json(res('scan_D.json'))
xs = np.array([r['xi'] for r in R]); DD = np.array([r['I'] for r in R]); aa = np.array([r['alpha'] for r in R])
zer = [r['zero'] for r in R]; nR = np.array([r['nR'] for r in R]); nD = np.array([r['nD'] for r in R])
osc = np.load(cache('oscill.npy'))   # xi, D, alpha, wall part of D, dalpha_wall, dalpha_Riemann, dalpha_dyadic (direct contour integral)
xo, Do, ao = osc[0], osc[1], osc[2]
zB = [x for x, z in zip(xs, zer) if z is not None]; zC = [x for x, n1, n2, z in zip(xs, nR, nD, zer) if n1+n2 > 0 and z is None]


def bands(ax):
    ax.axvspan(min(zB)-0.025, max(zB)+0.025, color=C4, alpha=0.13, lw=0)
    ax.axvspan(-8.2, max(x for x in zC if x < -3.5)+0.025, color=C7, alpha=0.08, lw=0)


# ---------- Fig. 1: D(rho) and alpha_D ----------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 7.2), sharex=True, gridspec_kw={'hspace': 0.08})
for a in (a1, a2): bands(a)
a1.semilogy(xo, Do, color=C1, lw=2, label='D(ρ) = 2(G(0) − G(ρ))')
a1.semilogy(xs[::4], DD[::4], 'o', ms=4.5, mfc='none', mec=INK, mew=0.9, label='thimble + Gauss–Hermite + Stokes')
a1.set_ylabel('D(ρ)'); a1.legend(loc='lower right')
a2.plot(xo, ao, color=C1, lw=2, label='α = d log D / d log ρ  (direct contour integral)')
a2.plot(xs[::4], aa[::4], 'o', ms=4.5, mfc='none', mec=INK, mew=0.9, label='thimble + Gauss–Hermite + Stokes')
a2.set_ylabel('α(log ρ)'); a2.set_xlabel('log ρ'); a2.set_xlim(-8.2, 8.2); a2.legend(loc='upper right')
a2.text(-7.9, 0.12, 'thimble traps the\nRiemann & dyadic walls', color=C7, fontsize=9)
a2.text(max(zB)+0.08, 1.15, 'thimble ends on a\nzero 6+iγₙ', color='#9a6a00', fontsize=9)
a2.text(0.3, 1.0, 'thimble bends left: nothing trapped;\nfor log ρ > −2.76 D is a convergent series\nin real powers of 1/ρ (no oscillation)', color=INK2, fontsize=9)
fig.savefig(f'{OUT}/fig1_D_alpha.png'); plt.close(fig)

# ---------- Fig. 3: Stokes staircases ----------
fig, ax = plt.subplots(figsize=(8.4, 4.2))
w = (xs > -5.0) & (xs < -2.5)
ax.step(xs[w], nR[w], where='mid', color=C7, label='trapped Riemann poles (of %d)' % len(GAMMA))
ax.step(xs[w], nD[w], where='mid', color=C2, label='trapped dyadic poles (of %d)' % len(DYAD))
zi = np.array([(z+1) if z is not None else 0 for z in zer])
ax.step(xs[w], zi[w], where='mid', color=C3, label='thimble ends on zero 6+iγₙ : n')
ax.set_xlabel('log ρ'); ax.set_ylabel('count / index'); ax.legend(loc='upper left', fontsize=9)
fig.savefig(f'{OUT}/fig3_staircase.png'); plt.close(fig)

# ---------- Fig. 4: thimbles in the q plane ----------
fig, ax = plt.subplots(figsize=(9.0, 7.6))
for xi, c in zip([1.0, -2.0, -2.8, -3.0, -3.5, -4.5, -6.0], [C1, C3, C4, C5, C2, C7, C6]):
    th = Thimble(xi, 'D'); t = np.linspace(th.eps, th.tend, 1500); p = th.q(t)
    P = np.concatenate([np.conj(p[::-1]), [th.q0], p])
    r = next(rr for rr in R if abs(rr['xi']-xi) < 1e-9)
    lab = 'log ρ = %.1f' % xi + ('  (ends on 6+iγ%d)' % (r['zero']+1) if r['zero'] is not None else '') + \
          ('  (traps %d R + %d D)' % (r['nR'], r['nD']) if r['nR']+r['nD'] > 0 and r['zero'] is None else '')
    ax.plot(P.real, P.imag, color=c, lw=1.8, label=lab); ax.plot([th.q0], [0], 'o', color=c, ms=6, mec=SURF, mew=1.5)
g = np.array(GAMMA); d = np.array(DYAD)
ax.plot(np.full(len(g)*2, 7.0), np.r_[g, -g], 'x', color=INK, ms=4, mew=1, label='poles 7 ± iγₙ, 15/2 ± 2πik/ln 2, real')
ax.plot(np.full(len(d)*2, 7.5), np.r_[d, -d], 'x', color=INK, ms=4, mew=1)
ax.plot([-1, 0, 2, 2.5, 4, 5.5, 6, 7.5, 8, 9.5, 10, 11.5, 12], np.zeros(13), 'x', color=INK, ms=5, mew=1.2)
ax.plot(np.full(len(g)*2, 6.0), np.r_[g, -g], 'o', mfc='none', mec=INK2, ms=4, mew=0.9, label='zeros 6 ± iγₙ, 13/2 + 2n')
ax.plot([6.5, 8.5, 10.5], [0, 0, 0], 'o', mfc='none', mec=INK2, ms=4, mew=0.9)
for x0 in (6, 7, 7.5): ax.axvline(x0, color=INK2, lw=0.8, ls=':')
ax.set_xlim(-12, 16); ax.set_ylim(-30, 30); ax.set_xlabel('Re q'); ax.set_ylabel('Im q')
ax.legend(loc='upper left', fontsize=8.2, ncol=1)
fig.savefig(f'{OUT}/fig4_thimbles.png'); plt.close(fig)

# ---------- log-periodic parts: spectrum vs structure function ----------
S = np.load(cache('spectrum_osc.npy')); edge = -LOGCMIN
sx, dnR, dnD = S[0], S[3], S[4]; m = sx >= edge+1.0
lr, daR, daD = osc[0], osc[5], osc[6]; k = lr <= -edge-0.5
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.semilogy(sx[m]-edge, np.abs(dnD[m]), color=C2, lw=1.6, label='energy spectrum, dyadic wall  (δ of d log H/d log κ)')
ax.semilogy(sx[m]-edge, np.abs(dnR[m]), color=C7, lw=1.6, label='energy spectrum, Riemann wall')
ax.semilogy(-edge-lr[k], np.abs(daD[k]), color=C2, lw=1.6, ls='--', label='structure function, dyadic wall  (δ of d log D/d log ρ)')
ax.semilogy(-edge-lr[k], np.abs(daR[k]), color=C7, lw=1.6, ls='--', label='structure function, Riemann wall')
ax.set_xlim(0.5, 5.5); ax.set_ylim(1e-30, 10)
ax.set_xlabel('distance Δ from the convergence edge:  Δ = log κ + log C_min  (spectrum),  Δ = −log ρ + log C_min  (structure function)', fontsize=8.5)
ax.set_ylabel('log-periodic part of the local index'); ax.legend(fontsize=8.3, loc='lower left')
fig.savefig(f'{OUT}/osc_compare.png', dpi=220); plt.close(fig)

# ---------- Fig. 10: what the finite width does to alpha ----------
fig, ax = plt.subplots(figsize=(8.4, 4.8))
x = np.linspace(-3, 4, 1400)
ax.plot(x, alphaL(x, 0.0)[0], color=INK, lw=2, label='infinite system')
for km, c in zip([3.0, 1.0, 0.3, 0.1], [C1, C3, C2, C7]):
    ax.plot(x, alphaL(x, km)[0], color=c, lw=1.5, label='κ_min = %.1f  (W/L = π/κ_min = %.1f)' % (km, np.pi/km))
ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-0.12, 0.6); ax.set_xlabel('log ρ'); ax.set_ylabel('α_W = d log D_W / d log ρ')
ax.legend(fontsize=9)
fig.savefig(f'{OUT}/fig10_finite_box_theory.png'); plt.close(fig)

if not have_mpi():
    print('Max Planck data not found: figures 9 and 13 skipped. Theory figures written to', OUT)
    raise SystemExit

# ---------- Fig. 9: per-run comparison, one physical width W for all runs ----------
W = 2.34      # metres: best common width of the Re_lambda <= 2398 runs (finite_box_W2.py)
F = load_json(cache('mpi_fit_box_full.json')); P = load_json(res('mpi_physical_units.json'))
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
fig.tight_layout(); fig.savefig(f'{OUT}/fig9_finite_box_runs.png', dpi=200); plt.close(fig)

# ---------- Fig. 13: extrapolation in 1/Re_lambda ----------
J = load_json(cache('mpi_extrap_invRe_full.json')); x = np.array(J['x']); Re = np.array(J['Re'])
B = np.array([[np.nan if v is None else v for v in row] for row in J['binned']])
Fi = J['fits']; A = Fi['Re>=1046 (adopted)']
ai, ae = np.array(A['ainf'], float), np.array(A['aerr'], float); s = A['s']
fig = plt.figure(figsize=(13, 7.6)); gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1], width_ratios=[1.35, 1], hspace=0.12, wspace=0.22)
ax = fig.add_subplot(gs[0, 0]); axr = fig.add_subplot(gs[1, 0], sharex=ax); axl = fig.add_subplot(gs[:, 1])
ok = np.isfinite(ai) & np.isfinite(ae)
ax.errorbar(x[ok], ai[ok], yerr=ae[ok], fmt='o', ms=5, color=INK, mfc=SURF, mew=1.3, elinewidth=1,
            label='data extrapolated to Re_λ → ∞ (in 1/Re_λ, runs Re_λ ≥ 1046)')
xx = np.linspace(-4, 2.7, 700)
ax.plot(xx, alphaL(xx-s, 0.0)[0], color=C1, lw=2.2, label='theory α_D(ρ), ρ = r/ℓ_D, ℓ_D/L = e^s = %.2f ± %.2f' % (np.exp(s), np.exp(s)*A['ds']))
for name, c in [('all 11 runs (Re>=413)', C3), ('Re>=2398', C2)]:
    ax.plot(xx, alphaL(xx-Fi[name]['s'], 0.0)[0], color=c, lw=1.2, ls='--', label='theory, s fitted to the %s extrapolation (s = %.2f)' % (
        'all-runs' if 'all' in name else 'Re_λ ≥ 2398', Fi[name]['s']))
ax.axhline(0, color=INK2, lw=0.8); ax.set_xlim(-4, 2.7); ax.set_ylim(-0.15, 0.8); ax.set_ylabel('α_∞ = d log S₂/d log r')
x_fit0 = x[(ai < 0.355) & (x > -3.0) & np.isfinite(ai)].min()
ax.axvspan(-4, x_fit0-0.05, color=GRID, alpha=0.5, lw=0); ax.text(-3.95, 0.02, 'not fitted: α_∞ > 0.355\n(inertial range and below)', fontsize=8.5, color=INK2)
ax.legend(fontsize=8.3, loc='upper right'); plt.setp(ax.get_xticklabels(), visible=False)
fit = (ai < 0.355) & (x > -3.0) & ok
axr.errorbar(x[fit], (ai-alphaL(x-s, 0.0)[0])[fit], yerr=ae[fit], fmt='o', ms=4, color=INK, mfc=SURF, mew=1.1, elinewidth=1)
axr.axhline(0, color=C1, lw=1.5); axr.set_ylim(-0.12, 0.12); axr.set_xlabel('log(r / L)'); axr.set_ylabel('α_∞ − α_D')
sel = Re >= 1000; X = 1/Re
for x0, c in zip([-1.0, 0.0, 0.5, 1.0, 1.5], [C7, C1, C3, C4, C2]):
    j = int(np.argmin(np.abs(x-x0))); y = B[sel, j]; o = np.isfinite(y)
    axl.plot(1e3*X[sel][o], y[o], 'o', color=c, ms=6, mec=SURF, mew=1.2)
    xl = np.linspace(0, 1e3*X[sel].max()*1.05, 20)
    axl.plot(xl, ai[j]+A['slope'][j]*xl/1e3, color=c, lw=1.4, label='log(r/L) = %.1f' % x0)
    axl.errorbar([0], [ai[j]], yerr=[ae[j]], fmt='s', color=c, ms=6, mfc=SURF, mew=1.5)
axl.set_xlim(-0.03, 1e3*X[sel].max()*1.08); axl.set_xlabel('10³ / Re_λ'); axl.set_ylabel('α at fixed r/L')
axl.legend(fontsize=8.5, loc='upper left')
fig.savefig(f'{OUT}/fig13_invRe_extrapolation.png'); plt.close(fig)
print('paper figures written to', OUT)
