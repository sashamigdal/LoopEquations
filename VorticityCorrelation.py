'''
CorrFunction[M_, T_, MySum_, WP_] :=
 Block[{\[Beta], p, q, f, sigmas, states},
  For[n = 1, n < M, n++,
   p = Prime[n];
   If[ 3 p < M, q = p, Break[]];
   ];
  p = RandomInteger[{1, q - 1}];
  \[Beta] = 2 Pi  p/q;
  Print[{\[Beta], p, q, M - q}];
  sigmas = Join[ Table[1, M - q], Table[-1, q]];
  corr =
   Table[
    {l,
      MySum[
       Block[{\[Alpha], F, \[Omega], II, Stot, sq},
        \[Alpha] = \[Beta] Accumulate[ RandomSample[sigmas]];
        F[k_] :=
         N[1/(2 Sin[\[Beta]/2]) {Cos[\[Alpha][[k]]],
            Sin[\[Alpha][[k]]], I Cos[\[Beta]/2]}, WP];
        \[Omega] =
         Chop[ I Cross[F[#], F[1 + Mod[# + 1, M]]] & /@ Range[M]];
        Stot = Sum[F[k], {k, M}];
        II[n_, m_] :=
         Block[{Snm, Smn, snm, smn, ds, k},
          Snm = Sum[F[k], {k, n, m}];
          Smn = Stot - Snm;
          snm = Snm/(m - n);
          smn = Smn/(n + M - m);
          ds = snm - smn ;
          sq = Exp[l] Sqrt[ds . ds];
          \[Omega][[n]] . \[Omega][[m]] 2/(M (M - 1)) Sin[sq ]/sq
          ];
        (*Print[II[1,5]];*)
        ans = Sum[II[n, m], {m, 2, M}, {n, 1, m - 1}];
        (*Print[{Exp[l],ans}];*)
        ans
        ], {t, T}]/T
     }
    , {l, -2, 0, 0.1}]
  ]



  \vec \omega_k = \frac {\sigma_k} {2} \cot \left (\frac {\beta } {2} \right)
  \left\{  \cos \left (\frac {\beta  \sigma_k } {2} + \alpha _k  \right),
   \sin \left (\frac {\beta  \sigma_k } {2} + \alpha _k \right),
    \imath \right\}


'''
import os
from plot import MakeDir, RankHistPos
# from functools import reduce
# from operator import add

from numpy import pi, sin, cos, tan, sqrt, exp
from sympy import prime
import numpy as np
from parallel import parallel_map, ConstSharedArray
import multiprocessing as mp

mp.set_start_method('fork')
def CorrFuncDir(M):
    return os.path.join("plots", "VorticityCorr." + str(M))





def corrfunction(M, T, R, l0, l1):
    MakeDir(CorrFuncDir(M))
    q = 1
    for n in range(1, M):
        p = prime(n)
        if 3 * p < M:
            q = p
        else:
            break
        pass
    p = np.random.randint(1, q - 1)
    beta = 2 * pi * p / q
    sigmas = ConstSharedArray(np.array([1] * (M - q) + [-1] * q, dtype=int))
    logrdata = np.linspace(l0, l1, R)
    rdata = ConstSharedArray(exp(logrdata))

    def SampleCorr(t):
        alphas = np.copy(sigmas[:])* beta
        np.random.shuffle(alphas)
        alphas = np.cumsum(alphas)
        alphas = np.append(alphas, alphas[0])

        def F(k):
            return 1 / (2 * sin(beta / 2)) * np.array([cos(alphas[k]), sin(alphas[k]), 1j * cos(beta / 2)], complex)

        def Omega(k):
            return (alphas[k + 1] - alphas[k]) / (2 * beta * tan(beta / 2)) * np.array(
                [cos((alphas[k] + alphas[k + 1]) / 2), sin((alphas[k] + alphas[k + 1]) / 2), 1j], complex)

        FF = np.array([F(k) for k in range(M)], complex)
        FS = np.cumsum(FF, axis=0)
        Stot = FS[-1]

        def II(n, m):
            Snm = FS[m] - FS[n]
            Smn = Stot - Snm
            snm = Snm / (m - n)
            smn = Smn / (n + M - m)
            ds = snm.real - smn.real
            sq = np.copy(rdata[:]) * sqrt(ds.dot(ds))
            return (Omega(n).dot(Omega(m))).real / (M * (M - 1)) * sin(sq) / sq

        ans = np.zeros((M, M, R), dtype=float),


        for m in range(1, M):
            for n in range(m):
                ans[n, m] = ans[m, n] = II(n, m)
            pass
        pass
        return np.sum(ans, axis=(0, 1))

    res = parallel_map(SampleCorr, range(T), mp.cpu_count())
    data = np.mean(np.vstack(res), axis=0)
    pathname = os.path.join(CorrFuncDir(M), "corrdata.np")
    data.tofile(pathname)
    return pathname


def test_corfunction():
    corrs = corrfunction(100, 100, 100, -5, 5)

def FDistribution(M,T):
    MakeDir(CorrFuncDir(M))
    q = 1
    for n in range(1, M):
        p = prime(n)
        if 3 * p < M:
            q = p
        else:
            break
        pass
    p = np.random.randint(1, q - 1)
    beta = 2 * pi * p / q
    sigmas = ConstSharedArray(np.array([1] * (M - q) + [-1] * q, dtype=int))
    def SampleCorr(t):
        alphas = np.copy(sigmas[:])* beta
        np.random.shuffle(alphas)
        alphas = np.cumsum(alphas)
        alphas = np.append(alphas, alphas[0])

        FF = 1 / (2 * sin(beta / 2)) * np.vstack([np.array([cos(alphas[k]), sin(alphas[k]), 1j * cos(beta / 2)], complex) for k in range(M)])
        FS = np.cumsum(FF, axis=0)
        Stot = FS[-1]
        m = np.random.randint(1,M)
        n = np.random.randint(0,m)
        Snm = FS[m] - FS[n]
        Smn = Stot - Snm
        snm = Snm / (m - n)
        smn = Smn / (n + M - m)
        return (snm.real - smn.real) ** 2
    res = parallel_map(SampleCorr, range(T), mp.cpu_count())
    data = np.array(res,float)
    pathname = os.path.join(CorrFuncDir(M), "Fdata.np")
    data.tofile(pathname)
    return pathname

def test_FDistribution():
    M = 1000
    T = 1000
    pathname = FDistribution(M, T)
    data = np.fromfile(pathname,float)
    plotpath = os.path.join(CorrFuncDir(M), "Fdata.png")
    RankHistPos(data,plotpath,name='SFHist',var_name='SF',logx=True, logy=True)