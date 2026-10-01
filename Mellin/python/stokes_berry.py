"""Berry-smoothed trapped-pole (log-periodic) part of the spectral index and of the structure-function index (Fig. 5).

Input: ../results/stokes_events.json (stokes_events.py). Use u = log kappa for the spectrum and u = -log rho for the
structure function, so that poles are trapped as u grows. At an event e the trapped set changes from A_e (u < u_e) to B_e,
and the thimble integral jumps by exactly minus the residue difference, J_e(u) = R[B_e](u) - R[A_e](u), so that H (or D)
stays continuous. Berry's smoothing of the jump [M. V. Berry, Proc. R. Soc. A 422, 7 (1989)] gives the trapped-pole part
    O(u) = sum_e S_e(u) J_e(u),   S_e = erfc(-(u - u_e)/w_e)/2,   w_e = sqrt(2 Re F_e)/|Im s_e|,
with the singulant F_e = W(p0) - W(s_e) between the main saddle and the secondary saddle s_e hit by the thimble at the
event (Im F_e = 0 mod 2 pi there; Im F_e(u) = -Im s_e (u - u_e) nearby). This is Eq. (S5.6) of Ref. [migdal2026Riemann]
with the pole singulant replaced by the saddle-connection one. The rest, H - O, is the smooth saddle (thimble) part, and
    delta n = n - d log(H - O)/du = (O' - n O)/(H - O),   n = H'/H,
an exact identity: no subtraction of nearly equal numbers. The same with D, alpha. Split into Riemann and dyadic poles (they
share the denominator, so the parts add up). The step version S_e = Theta(u - u_e) is the unsmoothed Stokes jump.
H, H' come from lineH.py (Re p = -1.5), and D, alpha from the line Re q = 1 (cache/lineD_c1.npy, as oscill.py).
Writes cache/stokes_berry.npz."""
import numpy as np
from scipy.special import erfc
from common import res, cache, load_json
import thimble_H as TH
import thimble as TD
from lineH import H_line, table

EV = load_json(res('stokes_events.json'))
NRD = {'H': (TH.NR, TH.ND), 'D': (len(TD.GAMMA), len(TD.DYAD))}


def coefficients(kind):
    """residue coefficients c_n and pole positions P_n of the trapped-pole term (contribution 2 Re c_n e^{P_n xi} for H,
    -2 Re c_n e^{q_n xi} for D, xi = log kappa or log rho)"""
    nR, nD = NRD[kind]
    out = {}
    for w, n in (('R', nR), ('D', nD)):
        c, P = [], []
        for k in range(n):
            if kind == 'H':
                r, p = TH.residue(w, k, 0.0); c.append(2*r); P.append(p)
            else:
                qn = 7+1j*TD.GAMMA[k] if w == 'R' else 7.5+1j*TD.DYAD[k]
                c.append(-2*TD.residue_at(qn, 0.0, 'D', w)); P.append(qn)
        out[w] = (np.array(c), np.array(P))
    return out


def trapped_part(kind, xi, sets, S):
    """O and dO/dxi of the poles in sets = {'R': weights, 'D': weights} (weights may be smooth functions of xi)"""
    C = coefficients(kind) if not hasattr(trapped_part, kind) else getattr(trapped_part, kind)
    setattr(trapped_part, kind, C)
    out = {}
    for w in ('R', 'D'):
        c, P = C[w]; wt = sets[w]                                   # wt: (len(xi), npoles)
        E = np.exp(np.outer(xi, P))*c
        O = (wt*E).sum(1).real; O1 = (wt*E*P).sum(1).real
        out[w] = (O, O1, S)
    return out


def singulant_track(e, u, h=0.01):
    """F_e on the grid u (sorted): s_e(xi) by Newton continuation in steps h from the event, dF/dxi = p0 - s_e"""
    kind = e['kind']; sx = 1 if kind == 'H' else -1
    if kind == 'H':
        Wp, W2, sad = (lambda p, xi: TH.dlogM(p)+xi), TH.d2logM, TH.saddle
    else:
        Wp, W2, sad = (lambda q, xi: TD.S1(q)+xi), TD.S2, (lambda xi: TD.saddle(xi, 'D'))
    ue, se = e['u'], complex(*e['saddle'])
    out = {}
    for direction, stop in ((+1, u.max()+h), (-1, u.min()-h)):
        uu = np.arange(ue, stop if direction > 0 else stop, direction*h)
        s = se; S = [se]; jumps = 0.0
        for x in uu[1:]:
            xi = sx*x; s_new = s
            for _ in range(30):
                ds = Wp(s_new, xi)/W2(s_new); s_new -= ds
                if abs(ds) < 1e-12: break
            jumps = max(jumps, abs(s_new-s)); s = s_new; S.append(s)
        S = np.array(S); P0 = np.array([sad(sx*x) for x in uu])
        g = (P0-S)*sx*direction*h                                       # dF = (p0 - s) dxi
        F = e['F'][0] + np.concatenate([[0], np.cumsum(0.5*(g[1:]+g[:-1]))])
        out[direction] = (uu, F, S, jumps)
    U = np.concatenate([out[-1][0][::-1], out[1][0][1:]]); F = np.concatenate([out[-1][1][::-1], out[1][1][1:]])
    Sd = np.concatenate([out[-1][2][::-1], out[1][2][1:]])
    Fi = np.interp(u, U, F.real)+1j*np.interp(u, U, F.imag)
    return Fi, U, F, Sd, max(out[1][3], out[-1][3])


def multiplier(e, u, mode):
    """Berry multiplier S_e(u): 'berry' (exact singulant), 'fixed' (linearised, fixed width w_e), 'step'"""
    if mode == 'step':
        return (u > e['u']).astype(float)
    if mode == 'fixed':
        return 0.5*erfc(-(u-e['u'])/e['width'])
    F = e['_F']
    sgn = np.sign(-(F.imag[np.searchsorted(e['_U'], e['u'])+5]-F.imag[np.searchsorted(e['_U'], e['u'])-5]))
    sig = -sgn*e['_Fu'].imag/np.sqrt(2*np.maximum(e['_Fu'].real, 1e-6))
    sig = np.where((e['_Fu'].real <= 1e-6) & (u < e['u']), np.inf, sig)  # untrapped side beyond Re F = 0
    return 0.5*erfc(-sig)


def weights(kind, u, mode='berry'):
    """per-pole multipliers sum_e S_e(u) (1[B_e] - 1[A_e]) on the grid u"""
    nR, nD = NRD[kind]
    wR = np.zeros((len(u), nR)); wD = np.zeros((len(u), nD))
    for e in EV['events']:
        if e['kind'] != kind: continue
        S = multiplier(e, u, mode)
        for k in e['B_R']: wR[:, k] += S
        for k in e['A_R']: wR[:, k] -= S
        for k in e['B_D']: wD[:, k] += S
        for k in e['A_D']: wD[:, k] -= S
    return {'R': wR, 'D': wD}


def index_shift(kind, u, mode='berry'):
    xi = u if kind == 'H' else -u
    if kind == 'H':
        Y, Mv = table(); F, F1 = H_line(xi, Y, Mv)
    else:
        line = np.load(cache('lineD_c1.npy')); y = line[0].real; Z = line[1]; q = 1.0+1j*y; h = y[1]-y[0]
        wS = np.full(len(y), 2.0); wS[1:-1:2] = 4.0; wS[0] = wS[-1] = 1.0; wS *= h/3/np.pi
        E = np.exp(np.outer(xi, q))*Z
        F = (E@wS).real; F1 = ((E*q)@wS).real
    n = F1/F
    parts = trapped_part(kind, xi, weights(kind, u, mode), None)
    O = parts['R'][0]+parts['D'][0]
    out = dict(F=F, n=n, O=O)
    for w in ('R', 'D'):
        Ow, Ow1 = parts[w][0], parts[w][1]
        out['d'+w] = (Ow1-n*Ow)/(F-O)
    out['dT'] = out['dR']+out['dD']
    out['O_rel'] = O/F
    # with respect to u: d/du = +d/dxi for H, -d/dxi for D (the index alpha = d log D/d log rho keeps its sign)
    return out


if __name__ == '__main__':
    first = {k: min(e['u'] for e in EV['events'] if e['kind'] == k) for k in ('H', 'D')}
    d = np.round(np.arange(-1.5, 2.5+1e-9, 0.002), 6)
    save = {'dist': d}
    from multiprocessing import Pool
    for kind in ('H', 'D'):
        u = first[kind]+d
        evs = [e for e in EV['events'] if e['kind'] == kind]
        with Pool(4) as p:
            tr = p.starmap(singulant_track, [(e, u) for e in evs], chunksize=1)
        for e, (Fu, U, F, Sd, jump) in zip(evs, tr):
            e['_Fu'], e['_U'], e['_F'] = Fu, U, F
            if e is evs[0] or kind == 'D':
                print('%s event u=%.6f: Re F %.2f -> %.2f / %.2f at distance -0.5 / +0.5 from it; saddle path %.3f%+.3fi .. %.3f%+.3fi'
                      ' (largest Newton step %.2f)' % (kind, e['u'], e['F'][0], np.interp(e['u']-0.5, U, F.real),
                      np.interp(e['u']+0.5, U, F.real), Sd[0].real, Sd[0].imag, Sd[-1].real, Sd[-1].imag, jump))
        for mode in ('berry', 'fixed', 'step'):
            r = index_shift(kind, u, mode)
            tag = kind+'_'+mode
            for k in ('dR', 'dD', 'dT', 'O_rel'):
                save[tag+'_'+k] = r[k]
            save[kind+'_n'] = r['n']
            i = np.argmax(np.abs(r['dT']))
            print('%s %-5s: max |delta index| = %.2e at distance %+.3f (index %.4f, rel. %.1e); Riemann part max %.1e, dyadic %.1e;'
                  ' max |O/%s| = %.1e; before the first trapping (distance < -0.5): max %.1e' % (
                      kind, mode, abs(r['dT'][i]), d[i], r['n'][i], abs(r['dT'][i]/r['n'][i]),
                      np.abs(r['dR']).max(), np.abs(r['dD']).max(), kind, np.abs(r['O_rel']).max(), np.abs(r['dT'][d < -0.5]).max()))
    save['events'] = np.array([[0 if e['kind'] == 'H' else 1, e['u']-first[e['kind']], e['width']] for e in EV['events']])
    np.savez(cache('stokes_berry.npz'), **save)
