"""Thimble + Gauss-Hermite + Stokes scan:  python scan.py D -8 8 0.05  ->  ../results/scan_D.json  (kind D or f)."""
import json, sys, time, numpy as np
from common import res
from multiprocessing import Pool
kind = sys.argv[1]; lo, hi, step = map(float, sys.argv[2:5])
def job(xi):
    from thimble import compute
    try:
        r = compute(float(xi), kind, 80)
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
    except Exception as e:
        return {'xi': float(xi), 'error': repr(e)}
if __name__ == '__main__':
    xis = np.round(np.arange(lo, hi+1e-9, step), 6)
    t = time.time()
    with Pool(4) as p:
        out = []
        for i, r in enumerate(p.imap(job, xis)):
            out.append(r)
            if i % 20 == 0: print(i, len(xis), r.get('xi'), r.get('I'), r.get('alpha'), round(time.time()-t), flush=True)
    json.dump(out, open(res(f'scan_{kind}.json'), 'w'))
    print('done', time.time()-t)
