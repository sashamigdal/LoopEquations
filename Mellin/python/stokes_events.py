"""Stokes events of the energy spectrum H(kappa) and of the structure function D(rho) on fine grids (Figs. 5, 6).

Variables: u = xi = log kappa for the spectrum (thimble_H.py, walls to the left) and u = -xi = -log rho for the structure
function (thimble.py, walls to the right), so that in both cases poles are trapped as u grows. An event is a value u_e where
the set of trapped poles changes, from A_e (u < u_e) to B_e. A thimble changes its topology only by passing through another
critical point: at u_e it runs into a secondary saddle s_e (W'(s_e) = 0) and the singulant F_e = W(p0) - W(s_e) has
Im F_e = 0 (mod 2 pi). The width of Berry's smoothing of the jump (stokes_berry.py) is w_e = sqrt(2 Re F_e)/|Im s_e|.

  1. spectrum: scan of log kappa on a 0.01 grid (thimble paths followed to t = TMAX, where the integrand has fallen by
     e^-TMAX^2), every change of the trapped set located by recursive bisection to 1e-8; secondary saddle at each event;
  2. structure function: the two events below log rho = -3.8 (scan_D.json), bisection to 1e-8, secondary saddles;
  3. checks at both sides of every event: thimble + residues (+ zero-parking tail) against the direct line integral
     (lineH.py, Re p = -1.5; cache/lineD_c1.npy, Re q = 1). Since both sides agree with the continuous line integral,
     the thimble jumps by exactly minus the residue jump.
Writes ../results/stokes_events.json (events, spectrum staircase, thimble paths at the first events)."""
import numpy as np
from multiprocessing import Pool
from common import res, cache, save_json

TMAX = 20.0                 # the default 12.4 gives the same trapped sets up to log kappa = 6.06 and misses poles beyond
XI_SCAN = (4.9, 6.8, 0.01)  # staircase (Fig. 6); events are located and smoothed for log kappa <= EVENT_MAX
EVENT_MAX = 6.3


# ---------------- states ----------------
def h_thimble(xi):
    import thimble_H as T
    return T.ThimbleH(xi, tmax=TMAX)


def h_state(xi):
    th = h_thimble(xi); iR, iD, z = th.trapped(); e = th.path[-1]
    return dict(u=float(xi), R=list(map(int, iR)), D=list(map(int, iD)), zero=z, end=[e.real, e.imag])


def d_state(u):
    from thimble import Thimble, GAMMA, DYAD
    th = Thimble(-u, 'D'); z = th.terminal_zero(); e = th.path[-1]
    return dict(u=float(u), R=list(map(int, th.trapped(GAMMA, 7.0, z is None))),
                D=list(map(int, th.trapped(DYAD, 7.5, z is None))), zero=z, end=[e.real, e.imag])


def key(st):
    return tuple(st['R']), tuple(st['D'])


def bisect_all(state, sa, sb, tol=1e-8):
    """all changes of the trapped set between the states sa and sb: list of (state before, state after)"""
    if key(sa) == key(sb):
        return []
    if sb['u']-sa['u'] <= tol:
        return [(sa, sb)]
    sm = state(0.5*(sa['u']+sb['u']))
    return bisect_all(state, sa, sm, tol)+bisect_all(state, sm, sb, tol)


# ---------------- secondary saddle and singulant ----------------
def secondary_saddle(kind, u):
    """the saddle met by the thimble at the event: Newton on W' = 0 from the point of the path (beyond 0.3 from p0) with
    the smallest |W'|; singulant F = W(p0) - W(s)"""
    if kind == 'H':
        import thimble_H as T
        xi = u; th = h_thimble(xi); P = th.dense(8000)
        Wp, W2 = (lambda p: T.dlogM(p)+xi), T.d2logM
        W = lambda p: T.logM(p)+xi*p; W0 = th.W0
    else:
        import mpmath as mm
        from thimble import Thimble, S1, S2, logZ
        xi = -u; th = Thimble(xi, 'D'); P = th._dense_path(6000)
        Wp, W2 = (lambda q: S1(q)+xi), S2
        W = lambda q: complex(logZ(q, 'D'))+xi*q; W0 = float(mm.re(logZ(th.q0, 'D')))+xi*th.q0
    a = np.array([abs(Wp(p)) for p in P]); k = int(np.argmin(np.where(np.abs(P-P[0]) > 0.3, a, np.inf))); s = P[k]
    for _ in range(50):
        ds = Wp(s)/W2(s); s = s-ds
        if abs(ds) < 1e-13: break
    F = W0-W(s)
    return complex(s), complex(F), float(a[k]), float(abs(Wp(s))), float(abs(s-P[k]))


# ---------------- checks against the line integrals ----------------
def check_H(xi):
    import thimble_H as T
    from lineH import H_line
    r = T.compute(xi, tmax=TMAX); Hl, H1l = H_line(xi)
    return dict(u=xi, rel_H=abs(r['H']/Hl[0]-1), abs_n=abs(r['n']-H1l[0]/Hl[0]), residues_rel=(r['SR']+r['SD'])/Hl[0],
                n=float(H1l[0]/Hl[0]))


def D_line(xi):
    line = np.load(cache('lineD_c1.npy')); y = line[0].real; Z = line[1]; q = 1.0+1j*y; h = y[1]-y[0]
    w = np.full(len(y), 2.0); w[1:-1:2] = 4.0; w[0] = w[-1] = 1.0; w *= h/3/np.pi
    E = np.exp(xi*q)*Z
    return float((E@w).real), float(((E*q)@w).real)


def check_D(u):
    from thimble import compute
    r = compute(-u, 'D'); Dl, D1l = D_line(-u)
    return dict(u=u, rel_D=abs(r['I']/Dl-1), abs_alpha=abs(r['alpha']-D1l/Dl), residues_rel=r['dI']/Dl, alpha=D1l/Dl)


# ---------------- jobs ----------------
def _bracket(args):
    kind, sa, sb = args
    return [(kind, a, b) for a, b in bisect_all(h_state if kind == 'H' else d_state, sa, sb)]


def _event(args):
    kind, a, b = args
    trap_new = set(b['R'])-set(a['R']) or set(b['D'])-set(a['D'])
    s, F, r0, r1, step = secondary_saddle(kind, b['u'])
    chk = [check_H(a['u']), check_H(b['u'])] if kind == 'H' else [check_D(a['u']), check_D(b['u'])]
    e = dict(kind=kind, u=0.5*(a['u']+b['u']), xi=0.5*(a['u']+b['u'])*(1 if kind == 'H' else -1), bracket=[a['u'], b['u']],
             A_R=a['R'], A_D=a['D'], B_R=b['R'], B_D=b['D'], zero_A=a['zero'], zero_B=b['zero'], end_A=a['end'], end_B=b['end'],
             saddle=[s.real, s.imag], F=[F.real, F.imag], ImF_mod_2pi=float((F.imag+np.pi) % (2*np.pi)-np.pi),
             width=float(np.sqrt(2*F.real)/abs(s.imag)), Wp_on_path=r0, Wp_newton=r1, newton_step=step, checks=chk)
    e['jump_rel'] = chk[1]['residues_rel']-chk[0]['residues_rel']
    return e


def _paths(args):
    kind, u = args
    if kind == 'H':
        P = h_thimble(u).dense(3000)
    else:
        from thimble import Thimble
        P = Thimble(-u, 'D')._dense_path(3000)
    return [P.real.tolist(), P.imag.tolist()]


def fmt(ks, n=6):
    ks = sorted(ks)
    return '%s%s' % ([k+1 for k in ks[:n]], '...' if len(ks) > n else '')


if __name__ == '__main__':
    lo, hi, st = XI_SCAN
    xs = np.round(np.arange(lo, hi+st/2, st), 6)
    with Pool(4) as p:
        scan = p.map(h_state, xs, chunksize=2)
        dsc = p.map(d_state, [3.80, 3.85, 4.20, 4.25])
    br = [('H', a, b) for a, b in zip(scan[:-1], scan[1:]) if key(a) != key(b) and b['u'] <= EVENT_MAX+1e-9]
    br += [('D', a, b) for a, b in zip(dsc[:-1], dsc[1:]) if key(a) != key(b)]
    print('%d spectrum and %d structure-function grid intervals with a change of the trapped set' % (
        sum(b[0] == 'H' for b in br), sum(b[0] == 'D' for b in br)))
    with Pool(4) as p:
        pairs = sum(p.map(_bracket, br, chunksize=1), [])
        events = sorted(p.map(_event, pairs, chunksize=1), key=lambda e: (e['kind'] != 'H', e['u']))
    for e in events:
        c = e['checks']
        dev = max(x['rel_H' if e['kind'] == 'H' else 'rel_D'] for x in c)
        print('%s u=%.8f: +R%s +D%s -R%s -D%s | saddle %.4f%+.4fi, F = %.3f%+.3fi (Im F mod 2pi %+.0e, |W\'| %.0e), width %.3f'
              ' | residue jump %.2e of the integral, thimble+residues vs line %.0e' % (
                  e['kind'], e['u'], fmt(set(e['B_R'])-set(e['A_R'])), fmt(set(e['B_D'])-set(e['A_D'])),
                  fmt(set(e['A_R'])-set(e['B_R'])), fmt(set(e['A_D'])-set(e['B_D'])), *e['saddle'], *e['F'],
                  e['ImF_mod_2pi'], e['Wp_newton'], e['width'], e['jump_rel'], dev))
    # checks on the grid, before the first trapping and in between the events
    with Pool(4) as p:
        gH = p.map(check_H, [4.9, 5.1, 5.3, 5.36, 5.4, 5.6, 5.8, 6.0, 6.2], chunksize=1)
        gD = p.map(check_D, [3.0, 3.5, 3.75, 3.85, 4.0, 4.3, 5.0, 6.0], chunksize=1)
    print('grid checks: spectrum max rel. dev. %.1e (index %.1e); structure function %.1e (alpha %.1e)' % (
        max(g['rel_H'] for g in gH), max(g['abs_n'] for g in gH), max(g['rel_D'] for g in gD), max(g['abs_alpha'] for g in gD)))
    first = {k: next(e for e in events if e['kind'] == k) for k in 'HD'}
    with Pool(4) as p:
        paths = p.map(_paths, [('H', first['H']['bracket'][0]), ('H', first['H']['bracket'][1])]
                      + [(e['kind'], e['bracket'][j]) for e in events if e['kind'] == 'D' for j in (0, 1)])
    stair = dict(u=[s['u'] for s in scan], nR=[len(s['R']) for s in scan], nD=[len(s['D']) for s in scan],
                 zero=[s['zero'] for s in scan], end=[s['end'] for s in scan], tmax=TMAX)
    save_json(dict(events=events, spectrum_staircase=stair, grid_checks=dict(H=gH, D=gD),
                   paths=dict(H=paths[:2], D=paths[2:])), res('stokes_events.json'), indent=1)
