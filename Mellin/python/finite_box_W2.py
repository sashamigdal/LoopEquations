import json, numpy as np
from scipy.optimize import minimize_scalar
from finite_box_W import fit_s, F, P
sub = [i for i, f in enumerate(F) if f['Re'] <= 2400]
N = sum(F[i]['n'] for i in sub)
obj = lambda W: np.sqrt(sum(fit_s(F[i], P[i]['eta'], W)[1]*F[i]['n'] for i in sub)/N)
r = minimize_scalar(obj, bounds=(1.2, 5), method='bounded', options={'xatol': 1e-3})
inf = np.sqrt(sum(F[i]['rms_inf']**2*F[i]['n'] for i in sub)/N); free = np.sqrt(sum(F[i]['rms_box']**2*F[i]['n'] for i in sub)/N)
print('runs Re<=2398: best W = %.3f m, pooled rms %.4f  (infinite %.4f, free k_min %.4f)' % (r.x, r.fun, inf, free))
# per-run implied W with a range: W where the per-run rms (s refitted) stays within 10%% of its minimum
out = []
for f, p in zip(F, P):
    Ws = np.exp(np.linspace(np.log(0.5), np.log(50), 61)); rm = np.array([np.sqrt(fit_s(f, p['eta'], W)[1]) for W in Ws])
    rinf = f['rms_inf']; j = int(np.argmin(rm)); ok = rm <= 1.1*rm[j]
    out.append(dict(Re=f['Re'], W_best=float(Ws[j]), W_lo=float(Ws[ok].min()), W_hi=float(Ws[ok].max()), rms_min=float(rm[j]), rms_inf=rinf,
                    open_ended=bool(ok[-1])))
    print('Re=%5.0f  W_best=%.2f m  10%%-range [%.2f, %s]  rms_min %.4f (inf %.4f)' % (f['Re'], Ws[j], Ws[ok].min(), ('inf' if ok[-1] else '%.2f' % Ws[ok].max()), rm[j], rinf))
json.dump({'W_sub': r.x, 'rms_sub': r.fun, 'rms_sub_inf': inf, 'rms_sub_free': free, 'perrun': out}, open('../mpi/fit_box_W2.json', 'w'), indent=1)
