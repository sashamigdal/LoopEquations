"""The computed figures of the paper, written to ../paper/figs (or the directory given as the first argument):

  fig1_D_alpha.png                 D(rho) and alpha(log rho): thimbles vs the direct contour integral
  fig3_staircase.png               Stokes staircases of the structure function
  fig4_thimbles.png                thimbles in the q plane
  osc_compare.png                  Berry-smoothed trapped-pole part of the spectral index and of alpha_D (Fig. 5)
  fig6_spectrum_staircase.png      Stokes staircase of the spectrum on a fine grid; thimbles at the first events (Fig. 6)
  fig10_finite_box_theory.png      alpha_W for several kappa_min
  fig9_finite_box_runs.png         Max Planck runs vs infinite system / per-run kappa_min / one width W   (needs the MPI data)
  fig13_invRe_extrapolation.png    1/Re_lambda extrapolation vs theory                                    (needs the MPI data)
  fig14_attractor_test.png         fixed-alpha cross-sections: turbulent attractor vs stochastization stage (needs the MPI data)

Inputs: ../results/scan_D.json (scan.py), cache/oscill.npy (oscill.py), ../results/stokes_events.json (stokes_events.py),
cache/stokes_berry.npz (stokes_berry.py),
cache/mpi_*_full.json (finite_box.py, physical_units.py, extrapolate_Re.py, extrap_invRe.py).
RegularPolygons.png and BSSpectra_clean.png are reproduced from Refs. [ReviewPaperAM, migdal2026Riemann]."""
import os, sys, numpy as np
from common import (cache, res, load_json, have_mpi, style, PAPER_FIGS, INK, INK2, GRID, SURF,
                    C1, C2, C3, C4, C5, C6, C7, SEQ, LOGCMIN, ALPHA_FIT, ALPHA_ATTR, ALPHA_ETA, RE_DECAYED)
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

# ---------- Fig. 5: Berry-smoothed trapped-pole part of the local index, from the first trapping on ----------
B = np.load(cache('stokes_berry.npz')); dist = B['dist']; evs = B['events']
EV = load_json(res('stokes_events.json'))
uc = {k: min(e['u'] for e in EV['events'] if e['kind'] == k) for k in 'HD'}
fig, axs = plt.subplots(2, 1, figsize=(7.8, 7.6), sharex=True, gridspec_kw={'hspace': 0.07})
for ax, kind, name in ((axs[0], 'H', '(a) energy spectrum:  δn,  n = d log H/d log κ'), (axs[1], 'D', '(b) structure function:  δα,  α = d log D/d log ρ')):
    ax.semilogy(dist, np.abs(B[kind+'_berry_dT']), color=INK, lw=2.4, label='trapped poles, Berry-smoothed')
    ax.semilogy(dist, np.abs(B[kind+'_berry_dD']), color=C2, lw=0.9, label='   dyadic poles')
    ax.semilogy(dist, np.abs(B[kind+'_berry_dR']), color=C7, lw=0.9, label='   Riemann poles')
    ax.semilogy(dist, np.abs(B[kind+'_step_dT']), color=INK2, lw=0.9, ls='--', label='unsmoothed Stokes jumps')
    ax.semilogy(dist, 0.01*np.abs(B[kind+'_n']), color=C1, lw=1.3, ls=':', label='1 % of the index')
    ev = evs[evs[:, 0] == (0 if kind == 'H' else 1)]
    ax.set_ylim(1e-16 if kind == 'H' else 1e-21, 3.0)
    ax.plot(ev[:, 1], np.full(len(ev), 1.2), 'v', color=INK, ms=4.5, mew=0, label='Stokes events (bisection)')
    ax.axvline(0, color=INK2, lw=0.6, ls=':')
    ax.text(0.015, 0.04, name, transform=ax.transAxes, fontsize=9.5, color=INK)
    ax.set_ylabel('|trapped-pole part of the index|')
axs[1].legend(fontsize=8.0, loc='upper right', bbox_to_anchor=(1.0, 0.86), ncol=2, framealpha=0.95)
axs[1].set_xlim(dist[0], dist[-1])
axs[1].set_xlabel('distance from the first trapping:  log κ − %.5f  (spectrum),   %.5f − log ρ  (structure function)' % (uc['H'], -uc['D']), fontsize=9)
fig.savefig(f'{OUT}/osc_compare.png', dpi=220); plt.close(fig)

# ---------- Fig. 6: Stokes staircase of the spectrum on a fine grid; thimbles at the first events ----------
st = EV['spectrum_staircase']; ug = np.array(st['u']); hev = [e for e in EV['events'] if e['kind'] == 'H']
dev = [e for e in EV['events'] if e['kind'] == 'D']
ue = [ug[0]] + [e['u'] for e in hev]; nRe = [0] + [len(e['B_R']) for e in hev]; nDe = [0] + [len(e['B_D']) for e in hev]
g = ug > hev[-1]['bracket'][1]
ue = np.r_[ue, ug[g]]; nRe = np.r_[nRe, np.array(st['nR'])[g]]; nDe = np.r_[nDe, np.array(st['nD'])[g]]
fig = plt.figure(figsize=(10.0, 9.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.25, wspace=0.2)
a = fig.add_subplot(gs[0, :])
a.step(ue, nRe+nDe, where='post', color=INK, lw=1.8, label='all trapped poles')
a.step(ue, nRe, where='post', color=C7, lw=1.4, label='Riemann wall  −8 + iγₙ')
a.step(ue, nDe, where='post', color=C2, lw=1.4, ls='--', label='dyadic wall  −17/2 + 2πim/log 2')
a.plot(ug, np.array(st['nR'])+np.array(st['nD']), 'o', ms=2.2, color=C3, mew=0, label='thimbles on the grid Δlog κ = 0.01')
a.axvline(hev[-1]['u'], color=INK2, lw=0.7, ls=':')
a.text(hev[-1]['u']+0.01, 2, 'events located by bisection to the left', fontsize=8, color=INK2)
a.set_xlim(ug[0], ug[-1]); a.set_xlabel('log κ'); a.set_ylabel('number of trapped poles'); a.legend(loc='upper left', fontsize=8.5)
a.text(0.015, 0.55, '(a)', transform=a.transAxes, fontsize=10)
gam = np.load(cache('zetazeros_120.npy')); dy = 2*np.pi*np.arange(1, 31)/np.log(2)
b1 = fig.add_subplot(gs[1, 0])
for (x, y), c, l in zip(EV['paths']['H'], [C1, C3], ['before: ends on −7+iγ₁', 'after: traps R1–R7, D1–D4']):
    b1.plot(x, y, color=c, lw=1.5, label=l)
b1.plot(np.full(len(gam), -8.0), gam, 'x', color=INK, ms=4, mew=1, label='poles')
b1.plot(np.full(len(dy), -8.5), dy, 'x', color=INK, ms=4, mew=1)
b1.plot(np.full(len(gam), -7.0), gam, 'o', mfc='none', mec=INK2, ms=4, mew=0.9, label='zeros −7+iγₙ')
b1.plot(*hev[0]['end_B'], 'D', mfc='none', mec=C3, ms=6, mew=1.2, label='f(p) = 0: end')
b1.plot(*hev[0]['saddle'], '*', color=INK, ms=11, mec=SURF, mew=0.6, label='secondary saddle')
for x0 in (-8, -8.5, -7): b1.axvline(x0, color=INK2, lw=0.6, ls=':')
b1.set_xlim(-16, 8); b1.set_ylim(0, 48); b1.set_xlabel('Re p'); b1.set_ylabel('Im p')
b1.legend(loc='center right', bbox_to_anchor=(1.0, 0.55), fontsize=7.0, framealpha=0.95)
b1.text(0.03, 0.03, '(b) spectrum, log κ₁ = %.7f' % hev[0]['u'], transform=b1.transAxes, fontsize=9)
b2 = fig.add_subplot(gs[1, 1])
labs = ['ρ₁ before: ends on 6+iγ₁', 'ρ₁ after: traps R, D m≥2', 'ρ₂ before: D1 free', 'ρ₂ after: D1 trapped']
for (x, y), c, l in zip(EV['paths']['D'], [C1, C3, C5, C7], labs):
    b2.plot(x, y, color=c, lw=1.5, label=l)
b2.plot(np.full(len(GAMMA), 7.0), GAMMA, 'x', color=INK, ms=4, mew=1, label='poles')
b2.plot(np.full(len(DYAD), 7.5), DYAD, 'x', color=INK, ms=4, mew=1)
b2.plot(np.full(len(GAMMA), 6.0), GAMMA, 'o', mfc='none', mec=INK2, ms=4, mew=0.9, label='zeros 6+iγₙ')
for e in dev: b2.plot(*e['saddle'], '*', color=INK, ms=11, mec=SURF, mew=0.6)
b2.plot([], [], '*', color=INK, ms=9, label='secondary saddles')
for x0 in (6, 7, 7.5): b2.axvline(x0, color=INK2, lw=0.6, ls=':')
b2.set_xlim(0, 16); b2.set_ylim(0, 30); b2.set_xlabel('Re q'); b2.set_ylabel('Im q')
b2.legend(loc='upper left', fontsize=6.8, framealpha=0.95)
b2.text(0.97, 0.97, '(c) structure function\nlog ρ₁ = %.7f\nlog ρ₂ = %.7f' % (dev[0]['xi'], dev[1]['xi']), transform=b2.transAxes,
        fontsize=8.5, ha='right', va='top')
fig.savefig(f'{OUT}/fig6_spectrum_staircase.png', dpi=200); plt.close(fig)

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

STOCH = '#d9d7d2'                 # stochastization stage (grey); attractor region left white
BOUND = '#b9a27c'                 # hatching of the boundary-effects region r > W
from matplotlib.patches import Patch
bound_patch = Patch(facecolor='none', edgecolor=BOUND, hatch='////', lw=0, label='boundary effects, r > W (excluded)')
# ---------- Fig. 9: per-run comparison, one physical width W for all runs ----------
W = round(load_json(res('mpi_fit_width_sub.json'))['W_sub'], 2)      # best common width of the Re_lambda <= 2398 runs (finite_box_W2.py)
F = load_json(cache('mpi_fit_box_full.json')); P = load_json(res('mpi_physical_units.json'))
fig, axs = plt.subplots(4, 3, figsize=(11, 12), sharey=True)
for ax, f, p in zip(axs.flat, F, P):
    lr, a, i0 = np.array(f['lr']), np.array(f['a']), f['i0']; X, A = lr[i0:], a[i0:]; eta = p['eta']
    e = lambda s: np.mean((A-alphaL(X-s, np.pi*eta*np.exp(s)/W)[0])**2)
    sW = minimize_scalar(e, bounds=(f['s_inf']-1.5, f['s_inf']+1.5), method='bounded').x
    x = np.linspace(lr[i0]-1.5, lr[-1]+0.4, 500)
    ax.axhspan(ALPHA_ATTR, 0.6, color=STOCH, alpha=0.6, lw=0)
    lW = np.log(W/eta); ax.axvspan(lW, lr[-1]+0.4, facecolor='none', edgecolor=BOUND, hatch='////', lw=0)
    ax.axvline(lW, color=BOUND, lw=1.0)
    att = a < ALPHA_ATTR
    ax.plot(lr[att], a[att], 'o', ms=3, mfc='none', mec=INK2, mew=0.7, label='Max Planck, turbulent attractor (α < %.3f)' % ALPHA_ATTR)
    ax.plot(lr[~att], a[~att], 'o', ms=3, mfc='none', mec='#aaa8a3', mew=0.7, label='Max Planck, stochastization stage')
    ax.plot(x, alphaL(x-f['s_inf'], 0.0)[0], color=INK2, lw=1.4, ls='--', label='infinite system')
    ax.plot(x, alphaL(x-f['s_box'], f['kmin'])[0], color=C1, lw=1.8, label='finite width, κ_min fitted per run')
    ax.plot(x, alphaL(x-sW, np.pi*eta*np.exp(sW)/W)[0], color=C2, lw=1.4, label='finite width, one W = %.2f m for all runs' % W)
    ax.axhline(0, color=GRID, lw=1); ax.axvline(lr[i0], color=INK2, lw=0.8, ls=':')
    ax.set_xlim(lr[i0]-1.5, lr[-1]+0.4); ax.set_ylim(-0.15, 0.6)
    ax.text(0.03, 0.05, 'Re_λ = %d\nrms %.3f → %.3f' % (round(f['Re']), f['rms_inf'], f['rms_box']), transform=ax.transAxes, fontsize=9)
    if f['Re'] < RE_DECAYED:
        ax.set_facecolor('#f1efe9')
        ax.text(0.97, 0.95, 'decayed turbulence\n(not in the Re_λ → ∞ fit)', transform=ax.transAxes, fontsize=8.5, color=INK2, ha='right', va='top')
    ax.set_xlabel('log(r/η)', fontsize=9)
axs.flat[-1].axis('off'); h, l = axs.flat[0].get_legend_handles_labels()
axs.flat[-1].legend(h+[bound_patch], l+['boundary effects, r > W = %.2f m' % W], loc='center', fontsize=9.5)
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
fit = (ai < ALPHA_FIT) & (x > -3.0) & ok
sto = ok & ~fit
x_att0 = x[fit].min()
xW0 = min(v for v, r in zip(J['x_W'], Re) if r >= RE_DECAYED)     # first adopted run to reach r = W
BB = np.array([[np.nan if v is None else v for v in row] for row in J['binned_boundary']])
ax.axvspan(-4, x_att0-0.05, color=STOCH, alpha=0.6, lw=0)
ax.text(-3.95, 0.72, 'stochastization stage (α_∞ ≥ %.3f)' % ALPHA_ATTR, fontsize=8.5, color=INK2)
ax.text(x_att0+0.06, 0.72, 'turbulent attractor', fontsize=8.5, color=INK2)
ax.axvspan(xW0, 2.7, facecolor='none', edgecolor=BOUND, hatch='////', lw=0); ax.axvline(xW0, color=BOUND, lw=1.0)
ax.text(xW0+0.04, 0.40, 'r > W\nin some\nruns', fontsize=8, color='#7d6a4a', bbox=dict(facecolor=SURF, edgecolor='none', pad=1.5))
for k in np.where(Re >= RE_DECAYED)[0]:
    ax.plot(x, BB[k], '.', ms=3, color=BOUND, label='single runs at r > W: boundary effects, excluded' if k == np.argmax(Re >= RE_DECAYED) else None)
ax.errorbar(x[fit], ai[fit], yerr=ae[fit], fmt='o', ms=5, color=INK, mfc=INK, mew=1.3, elinewidth=1,
            label='Re_λ → ∞ (in 1/Re_λ, runs Re_λ ≥ 1046): turbulent attractor, fitted')
ax.errorbar(x[sto], ai[sto], yerr=ae[sto], fmt='o', ms=5, color='#8f8d88', mfc=SURF, mew=1.1, elinewidth=1,
            label='Re_λ → ∞: stochastization stage, not fitted')
xx = np.linspace(-4, 2.7, 700)
ax.plot(xx, alphaL(xx-s, 0.0)[0], color=C1, lw=2.2, label='theory α_D(ρ), ρ = r/ℓ_D, ℓ_D/L = e^s = %.2f ± %.2f' % (np.exp(s), np.exp(s)*A['ds']))
for name, c in [('all 11 runs (Re>=413)', C3), ('Re>=2398', C2)]:
    ax.plot(xx, alphaL(xx-Fi[name]['s'], 0.0)[0], color=c, lw=1.2, ls='--', label='theory, s from the %s (s = %.2f)' % (
        'all 11 runs' if 'all' in name else 'runs Re_λ ≥ 2398', Fi[name]['s']))
ax.axhline(0, color=INK2, lw=0.8); ax.set_xlim(-4, 2.7); ax.set_ylim(-0.15, 0.8); ax.set_ylabel('α_∞ = d log S₂/d log r')
ax.legend(fontsize=7.8, loc='lower left', frameon=True, facecolor=SURF, edgecolor='none', framealpha=0.9); plt.setp(ax.get_xticklabels(), visible=False)
axr.axvspan(-4, x_att0-0.05, color=STOCH, alpha=0.6, lw=0)
axr.axvspan(xW0, 2.7, facecolor='none', edgecolor=BOUND, hatch='////', lw=0); axr.axvline(xW0, color=BOUND, lw=1.0)
axr.errorbar(x[fit], (ai-alphaL(x-s, 0.0)[0])[fit], yerr=ae[fit], fmt='o', ms=4, color=INK, mfc=INK, mew=1.1, elinewidth=1)
axr.axhline(0, color=C1, lw=1.5); axr.set_ylim(-0.12, 0.12); axr.set_xlabel('log(r / L)'); axr.set_ylabel('α_∞ − α_D')
sel = Re >= RE_DECAYED; X = 1/Re
for x0, c in zip([-1.0, 0.0, 0.5, 1.0, 1.5], [C7, C1, C3, C4, C2]):
    j = int(np.argmin(np.abs(x-x0))); y = B[sel, j]; o = np.isfinite(y)
    axl.plot(1e3*X[sel][o], y[o], 'o', color=c, ms=6, mec=SURF, mew=1.2)
    yd = B[~sel, j]; od = np.isfinite(yd)
    axl.plot(1e3*X[~sel][od], yd[od], 'x', color='#aaa8a3', ms=6, mew=1.3, label='decayed turbulence (Re_λ < 10³, not used)' if x0 == -1.0 else None)
    xl = np.linspace(0, 1e3*X[sel].max()*1.05, 20)
    axl.plot(xl, ai[j]+A['slope'][j]*xl/1e3, color=c, lw=1.4, label='log(r/L) = %.1f' % x0)
    axl.errorbar([0], [ai[j]], yerr=[ae[j]], fmt='s', color=c, ms=6, mfc=SURF, mew=1.5)
axl.set_xlim(-0.03, 1e3*X.max()*1.05); axl.set_xlabel('10³ / Re_λ'); axl.set_ylabel('α at fixed r/L')
axl.legend(fontsize=8.5, loc='upper left')
fig.savefig(f'{OUT}/fig13_invRe_extrapolation.png'); plt.close(fig)

# ---------- Fig. 14: turbulent attractor vs stochastization stage (fixed-alpha cross-sections) ----------
T = load_json(res('mpi_attractor_region.json')); C = load_json(cache('attractor_crossings.json'))
lev = np.array(T['levels']); rows = T['rows']; ref = T['ref_level']; a_star = ALPHA_ATTR
Xe = np.array([[np.nan if v is None else v for v in row] for row in C['log_r_over_eta']]); kref = list(lev).index(ref)
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={'wspace': 0.28, 'width_ratios': [1.25, 1, 1]})
k_hi = 0
for k, run in enumerate(C['runs']):
    lr, a = np.array(run['lr']), np.array(run['a']); iW = lr <= run['lr_W']
    if run['Re'] < RE_DECAYED:
        a1.plot((lr-Xe[k, kref])[iW], a[iW], '--', lw=1.0, color='#aaa8a3', label='Re_λ = %d (decayed)' % round(run['Re']))
    else:
        a1.plot((lr-Xe[k, kref])[iW], a[iW], '-', lw=1.1, color=SEQ[3+k_hi], label='Re_λ = %d' % round(run['Re'])); k_hi += 1
    a1.plot((lr-Xe[k, kref])[~iW], a[~iW], '-', lw=0.8, color=BOUND, label='r > W (boundary effects, excluded)' if k == len(C['runs'])-1 else None)
a1.axhspan(a_star, 2.1, color=STOCH, alpha=0.6, lw=0); a1.axhline(ALPHA_ETA, color=INK2, lw=0.8, ls=':')
a1.text(-6.8, 1.05, 'stochastization stage', fontsize=9, color=INK2); a1.text(-6.8, 0.12, 'turbulent attractor', fontsize=9, color=INK2)
a1.text(-6.8, ALPHA_ETA+0.03, 'above: r_α follows η', fontsize=8, color=INK2)
a1.set_xlim(-7, 3); a1.set_ylim(-0.1, 2.05); a1.set_xlabel('log(r / r₀.₃),  r₀.₃: where α = 0.3 in each run')
a1.set_ylabel('α = d log S₂/d log r'); a1.legend(fontsize=6.6, ncol=2, loc='upper right')
sc = np.array([r['scatter'] for r in rows]); sca = np.array([r['scatter_all_runs'] for r in rows]); m = lev != ref
a2.axvspan(a_star, 1.25, color=STOCH, alpha=0.6, lw=0); a2.axvline(ALPHA_ETA, color=INK2, lw=0.8, ls=':')
a2.semilogy(lev[m], sc[m], 'o-', color=INK, ms=5, lw=1.2, label='runs Re_λ ≥ 1046')
a2.semilogy(lev[m], sca[m], 's--', color='#aaa8a3', ms=4, lw=1, label='including decayed runs')
a2.axhline(T['tolerance'], color=INK2, lw=0.8, ls='--'); a2.axhline(T['tolerance_loose'], color=INK2, lw=0.8, ls=':'); a2.set_xlim(0.05, 1.25)
a2.set_xlabel('α'); a2.set_ylabel('scatter of log(r_α / r₀.₃) across runs'); a2.legend(fontsize=8.5, loc='upper left')
be = np.array([r['slope_eta'] for r in rows]); bee = np.array([r['slope_eta_err'] for r in rows])
br = np.array([r['slope_ref'] for r in rows]); bre = np.array([r['slope_ref_err'] for r in rows])
a3.axvspan(a_star, 1.25, color=STOCH, alpha=0.6, lw=0); a3.axvline(ALPHA_ETA, color=INK2, lw=0.8, ls=':'); a3.axhline(0, color=INK2, lw=0.8)
a3.errorbar(lev, be, yerr=bee, fmt='o-', color=C7, ms=4, lw=1.1, label='r_α in units of the Kolmogorov length η')
a3.errorbar(lev, br, yerr=bre, fmt='s-', color=C1, ms=4, lw=1.1, label='r_α in units of the attractor length r₀.₃')
a3.set_xlim(0.05, 1.25); a3.set_xlabel('α'); a3.set_ylabel('d log r_α / d log Re_λ  (runs Re_λ ≥ 1046)'); a3.legend(fontsize=8.2, loc='center left')
fig.savefig(f'{OUT}/fig14_attractor_test.png'); plt.close(fig)
print('paper figures written to', OUT)
