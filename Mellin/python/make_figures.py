"""Figures for the MellinOscillationsOdd thimble computation (writes PNGs into ../results)."""
import json, os, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thimble import Thimble, GAMMA, DYAD

OUT = sys.argv[1] if len(sys.argv) > 1 else '../results'
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
C1, C2, C3, C4, C5, C6, C7 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7'
SEQ = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b', '#0a2a55']
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'lines.linewidth': 2, 'font.size': 10.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})

R = json.load(open('scan_D.json')); RF = json.load(open('scan_f.json')) if os.path.exists('scan_f.json') else None
xs = np.array([r['xi'] for r in R]); DD = np.array([r['I'] for r in R]); aa = np.array([r['alpha'] for r in R])
zer = [r['zero'] for r in R]; nR = np.array([r['nR'] for r in R]); nD = np.array([r['nD'] for r in R])
osc = np.load('oscill.npy')           # xi, D, alpha, wall part of D, dalpha_wall, dalpha_Riemann, dalpha_dyadic (exact line integral)
xo, Do, ao = osc[0], osc[1], osc[2]
zB = [x for x, z in zip(xs, zer) if z is not None]; zC = [x for x, n1, n2, z in zip(xs, nR, nD, zer) if n1+n2 > 0 and z is None]
LOGCMIN, LOGCMAX = -2.9153, -2.7635   # log of min/max C(Delta)


def bands(ax, ylab=None):
    ax.axvspan(min(zB)-0.025, max(zB)+0.025, color=C4, alpha=0.13, lw=0)
    ax.axvspan(-8.2, max(x for x in zC if x < -3.5)+0.025, color=C7, alpha=0.08, lw=0)


# ---------- 1. D(r) and alpha_D ----------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 7.2), sharex=True, gridspec_kw={'hspace': 0.08})
for a in (a1, a2): bands(a)
a1.semilogy(xo, Do, color=C1, lw=2, label='D(r) = 2(f(0) − f(r))')
a1.semilogy(xs[::4], DD[::4], 'o', ms=4.5, mfc='none', mec=INK, mew=0.9, label='thimble + Gauss–Hermite + Stokes')
a1.set_ylabel('D(r)'); a1.legend(loc='lower right')
a2.plot(xo, ao, color=C1, lw=2, label='α = d log D / d log r  (direct line integral)')
a2.plot(xs[::4], aa[::4], 'o', ms=4.5, mfc='none', mec=INK, mew=0.9, label='thimble + Gauss–Hermite + Stokes')
a2.set_ylabel('α(log r)'); a2.set_xlabel('log r'); a2.set_xlim(-8.2, 8.2); a2.legend(loc='upper right')
a2.text(-7.9, 0.12, 'thimble traps the\nRiemann & dyadic walls', color=C7, fontsize=9)
a2.text(max(zB)+0.08, 1.15, 'thimble ends on a\nzero 6+iγₙ', color='#9a6a00', fontsize=9)
a2.text(0.3, 1.0, 'thimble bends left: nothing trapped;\nfor log r > −2.76 D is a convergent series\nin real powers of 1/r (no oscillation)', color=INK2, fontsize=9)
err = np.max(np.abs(DD/np.interp(xs, xo, Do)-1)); erra = np.max(np.abs(aa-np.interp(xs, xo, ao)))
a1.set_title('Structure function D(r) and its index — thimbles vs direct integral (max rel. dev. %.0e, |Δα| ≤ %.0e)' % (err, erra), fontsize=10.5, loc='left')
fig.savefig(f'{OUT}/fig1_D_alpha.png'); plt.close(fig)

# ---------- 2. oscillating part of alpha ----------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 7.6), gridspec_kw={'hspace': 0.42})
ok = xo < LOGCMIN
a1.semilogy(xo[ok], np.abs(osc[6][ok]), color=C2, lw=1.6, label='dyadic poles 15/2 + 2πik/ln 2')
a1.semilogy(xo[ok], np.abs(osc[5][ok]), color=C7, lw=1.6, label='Riemann poles 7 + iγₙ')
st = np.array([(r['xi'], abs(r['dalpha_stokes'])) for r in R if r['dalpha_stokes'] != 0])
a1.semilogy(st[:, 0], st[:, 1], 'o', ms=4, mfc='none', mec=INK, mew=0.9, label='Stokes terms picked up by the thimbles')
a1.axvline(LOGCMIN, color=INK2, ls='--', lw=1); a1.text(LOGCMIN+0.05, 1e-30, 'log C_min: residue\nseries diverges →', fontsize=8.5, color=INK2)
a1.set_xlim(-8.2, -2.4); a1.set_ylim(1e-40, 1e-4); a1.set_xlabel('log r'); a1.set_ylabel('|δα| (log-periodic part)')
a1.legend(loc='upper left', fontsize=9); a1.set_title('Oscillating part of α_D: all wall poles (lines) vs Stokes residues of trapped poles (circles)', fontsize=10.5, loc='left')
m = (xo > -7.5) & (xo < -3.0)
env = np.exp(5.5*(xo[m]+3.0))
a2.plot(xo[m], osc[4][m]/env, color=C2, lw=1.6)
a2.set_xlabel('log r'); a2.set_ylabel('δα / e^{5.5(log r + 3)}')
a2.set_title('Same, divided by its r^{5.5} envelope: period ln 2 = 0.693 in log r (dyadic k = 1 dominates)', fontsize=10.5, loc='left')
fig.savefig(f'{OUT}/fig2_oscillation.png'); plt.close(fig)

# ---------- 3. Stokes staircase ----------
fig, ax = plt.subplots(figsize=(8.4, 4.2))
w = (xs > -5.0) & (xs < -2.5)
ax.step(xs[w], nR[w], where='mid', color=C7, label='trapped Riemann poles (of %d)' % len(GAMMA))
ax.step(xs[w], nD[w], where='mid', color=C2, label='trapped dyadic poles (of %d)' % len(DYAD))
zi = np.array([(z+1) if z is not None else 0 for z in zer])
ax.step(xs[w], zi[w], where='mid', color=C3, label='thimble ends on zero 6+iγₙ : n')
ax.set_xlabel('log r'); ax.set_ylabel('count / index'); ax.legend(loc='upper left', fontsize=9)
ax.set_title('Stokes staircases: wall trapping (log r < −3.85) and zero termination (−3.8 ≤ log r ≤ −2.95)', fontsize=10.5, loc='left')
fig.savefig(f'{OUT}/fig3_staircase.png'); plt.close(fig)

# ---------- 4. thimbles in the q plane ----------
fig, ax = plt.subplots(figsize=(9.0, 7.6))
sel = [1.0, -2.0, -2.8, -3.0, -3.5, -4.5, -6.0]; cols = [C1, C3, C4, C5, C2, C7, C6]
for xi, c in zip(sel, cols):
    th = Thimble(xi, 'D'); t = np.linspace(th.eps, th.tend, 1500); p = th.q(t)
    P = np.concatenate([np.conj(p[::-1]), [th.q0], p])
    r = next(rr for rr in R if abs(rr['xi']-xi) < 1e-9)
    lab = 'log r = %.1f' % xi + ('  (ends on 6+iγ%d)' % (r['zero']+1) if r['zero'] is not None else '') + ('  (traps %d R + %d D)' % (r['nR'], r['nD']) if r['nR']+r['nD'] > 0 and r['zero'] is None else '')
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
ax.set_title('Lefschetz thimbles of r^q(−2ZR(q)) through the real saddles q₀ ∈ (0, 2)', fontsize=10.5, loc='left')
fig.savefig(f'{OUT}/fig4_thimbles.png'); plt.close(fig)

# ---------- 5. cutoff artifact ----------
cut = np.load('xi2_cut.npy')
fig, ax = plt.subplots(figsize=(8.4, 4.4))
L = np.linspace(0.5, 2.5, 400)
ax.plot(L, np.interp(L*np.log(10), xo, ao), color=C1, label='exact α_D (Mellin integral, no cutoff)')
ax.plot(cut[0], cut[1], color=C2, lw=1.6, label='old xi2: ∫ over 0.1 < k < 1000 of htab (CorrelationOscillation.nb)')
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlim(1.0, 2.5); ax.set_ylim(-0.0008, 0.0065); ax.set_xlabel('log10 r'); ax.set_ylabel('effective index')
ax.legend(loc='upper right', fontsize=9)
ax.set_title('The large-r oscillations of the old index come from the k-integration cutoff at k = 0.1', fontsize=10.5, loc='left')
fig.savefig(f'{OUT}/fig5_cutoff_artifact.png'); plt.close(fig)

# ---------- 6. Max Planck comparison ----------
fits = json.load(open('../mpi/fit_tail.json'))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 8.2), gridspec_kw={'hspace': 0.25, 'height_ratios': [2.2, 1]})
xx = np.linspace(-8, 3, 800)
for k, f in enumerate(fits):
    lr, a = np.array(f['lr']), np.array(f['a'])
    a1.plot(lr-f['shift'], a, '.', ms=3.2, color=SEQ[k], label='Re_λ = %d' % round(f['Re']))
    i0 = f['i0']; X = lr[i0:]-f['shift']
    a2.plot(X, a[i0:]-np.interp(X, xo, ao), '.-', ms=3, lw=0.6, color=SEQ[k])
a1.plot(xx, np.interp(xx, xo, ao), color=INK, lw=2.2, label='theory α_D')
a1.set_xlim(-8, 3); a1.set_ylim(-0.15, 2.1); a1.set_xlabel('log(r/η) − s'); a1.set_ylabel('α = d log S₂ / d log r')
a1.legend(loc='upper right', fontsize=8, ncol=2)
a1.set_title('Max Planck wind tunnel (E_Kohler): measured index, each run shifted by its fitted s', fontsize=10.5, loc='left')
a2.axhline(0, color=INK2, lw=0.8); a2.set_xlim(-2.2, 0.8); a2.set_xlabel('log(r/η) − s  (fitted tail, α_exp < 0.355)'); a2.set_ylabel('α_exp − α_D')
fig.savefig(f'{OUT}/fig6_maxplanck.png'); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.0, 4.0))
Re = np.array([f['Re'] for f in fits]); S = np.array([f['shift'] for f in fits]); pf = np.polyfit(np.log(Re), S, 1)
ax.plot(np.log(Re), S, 'o', color=C1, ms=7, mec=SURF, mew=1.5)
xl = np.linspace(np.log(Re).min()-0.2, np.log(Re).max()+0.2, 10)
ax.plot(xl, np.polyval(pf, xl), color=INK2, lw=1.2, ls='--', label='slope %.2f  (L/η ∝ Re_λ^{3/2} ⇒ 1.5)' % pf[0])
ax.set_xlabel('log Re_λ'); ax.set_ylabel('fitted shift s = log(r/η) − log r_theory'); ax.legend(loc='upper left', fontsize=9)
fig.savefig(f'{OUT}/fig7_shift_vs_Re.png'); plt.close(fig)

# ---------- 8. f(r) ----------
if RF:
    xf = np.array([r['xi'] for r in RF]); ff = np.array([r['I'] for r in RF]); af = np.array([r['alpha'] for r in RF])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 6.6), sharex=True, gridspec_kw={'hspace': 0.08})
    a1.semilogy(xf, ff, color=C3); a1.set_ylabel('f(r) = ∫ H(k) sin(kr)/(kr) dk')
    a2.plot(xf, af, color=C3); a2.set_ylabel('α_f = d log f / d log r'); a2.set_xlabel('log r')
    a1.set_title('Correlation function f(r) and its index (thimble + Gauss–Hermite + Stokes)', fontsize=10.5, loc='left')
    fig.savefig(f'{OUT}/fig8_f_alpha.png'); plt.close(fig)
print('figures written to', OUT)
