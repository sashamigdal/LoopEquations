"""Tabulate ABC(Delta) on Chebyshev nodes of [Delta1, Delta2]; r2 by continuation from Delta=0.5 as in the .wl."""
import json, os, sys, time
from multiprocessing import Pool
from mpmath import mp, mpf, cos, pi
from abcport import r2, IQ, action, alal, cons, re, D1, D2
mp.dps = 20
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
nodes = [ (D1+D2)/2 + (D2-D1)/2*cos(pi*(j+mpf(1)/2)/N) for j in range(N) ]   # Chebyshev points (1st kind)
nodes = sorted(nodes, reverse=True)
# continuation of the root r2(D) from D=0.5 downward in steps <= 0.001
rr = mpf('44.74657089612265'); d = mpf('0.5'); rvals = []
for x in nodes:
    while d - x > mpf('0.001'):
        d -= mpf('0.001'); rr = r2(d, rr)
    d = x; rr = r2(d, rr); rvals.append(rr)
def work(args):
    x, r = args
    L = re(action(r, x)); S = re(alal(x, r)); J = re(cons(x, r)); iq = IQ(x, r)
    return [str(x), str(r), str(L), str(S), str(J), str(iq)]
if __name__ == '__main__':
    t = time.time()
    with Pool(3) as p:
        rows = p.map(work, list(zip(nodes, rvals)))
    json.dump({'D1': str(D1), 'D2': str(D2), 'cols': ['D','r','L','S','J','IQ'], 'rows': rows}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'abc_cheb{N}.json'), 'w'), indent=0)
    print('done', N, time.time()-t)
