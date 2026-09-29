import numpy as np, glob, re, json
from scipy.optimize import minimize_scalar
o=np.load('py/oscill.npy'); d=(o[0], o[1], o[2])    # exact alpha_D(xi) on a 0.01 grid
aD=lambda x: np.interp(x, d[0], d[2])
num=lambda pat, f: float(re.search(pat, f.split('/')[-1]).group(1).rstrip('.'))
files=sorted(glob.glob('mpi/E_Kohler/Re_*.csv'), key=lambda f: num(r'Re_([0-9.]+)_', f))
out=[]
for f in files:
    Re=num(r'Re_([0-9.]+)_', f); eps=num(r'Eps_([0-9.]+)\.csv', f)
    dd=np.genfromtxt(f, delimiter=',', skip_header=1); r,S2=dd[:,1],dd[:,2]
    lr=np.log(r); lS=np.log(S2)
    a=np.empty_like(lr); a[1:-1]=(lS[2:]-lS[:-2])/(lr[2:]-lr[:-2]); a[0]=(lS[1]-lS[0])/(lr[1]-lr[0]); a[-1]=(lS[-1]-lS[-2])/(lr[-1]-lr[-2])
    i0=np.argmax((a<0.355)&(lr>lr.mean()))       # tail window as in CorrelationOscillation.nb
    X, A = lr[i0:], a[i0:]
    err=lambda s: np.mean((A-aD(X-s))**2)
    res=minimize_scalar(err, bounds=(0, 20), method='bounded')
    s=res.x; rms=np.sqrt(res.fun); resid=A-aD(X-s)
    out.append(dict(lr=lr.tolist(), a=a.tolist(), i0=int(i0), file=f.split('/')[-1], Re=Re, eps=eps, shift=s, rms=rms, n=int(len(X)), logr_range=[float(X[0]),float(X[-1])]))
    print('Re_lambda=%6.0f  shift ln(r/eta)-xi = %6.3f (log10 %.3f)  rms=%.4f  n=%d  |resid|max=%.3f' % (Re, s, s/np.log(10), rms, len(X), np.abs(resid).max()))
json.dump(out, open('mpi/fit_tail.json','w'), indent=1)
R=np.array([o['Re'] for o in out]); S=np.array([o['shift'] for o in out])
p=np.polyfit(np.log(R), S, 1); print('shift vs ln Re_lambda: slope %.3f (L/eta ~ Re_lambda^1.5 -> 1.5), intercept %.3f' % tuple(p))
