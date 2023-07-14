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

from mpmath import rational

from plot import MakeDir, RankHistPos, RankHist2, XYPlot
# from functools import reduce
# from operator import add

from numpy import pi, sin, cos, tan, sqrt, exp
import numpy as np
from parallel import ConstSharedArray
import multiprocessing as mp
import concurrent.futures as fut
from fractions import Fraction
def CorrFuncDir(M):
    return os.path.join("plots", "VorticityCorr." + str(M))


def F(k, alphas, beta):
    return 1 / (2 * sin(beta / 2)) * np.array([cos(alphas[k]), sin(alphas[k]), 1j * cos(beta / 2)], complex)


def Omega(k, alphas, beta):
    return (alphas[k + 1] - alphas[k]) / (2 * beta * tan(beta / 2)) * np.array(
        [cos((alphas[k] + alphas[k + 1]) / 2), sin((alphas[k] + alphas[k + 1]) / 2), 1j], complex)


class FDPlotter():
    def __init__(self, M, T, R):
        self.M = M
        self.T = T
        self.R = R
        if not os.path.isfile(self.FDistributionPathname()):
            self.FDistribution()
            print("made FDistribution " + str(M) )
        data = None
        try:
            data = np.fromfile(self.FDistributionPathname(),float).reshape(-1,3).T
            plotpath = os.path.join(CorrFuncDir(M), "beta.png")
            RankHistPos(data[0],plotpath,name='beta Hist',var_name='beta',logx=True, logy=True)
            print("made betaHist " + str(M) )
            plotpath = os.path.join(CorrFuncDir(M), "DS.png")
            RankHistPos(data[1],plotpath,name='DSHist',var_name='DS',logx=True, logy=True)
            print("made DS Hist " + str(M) )
            plotpath = os.path.join(CorrFuncDir(M), "OmegaOmega.png")
            RankHistPos(-data[2], plotpath, name='OmegaOmega Hist', var_name='OdotO', logx=True, logy=True)
            print("made OmegaOmega " + str(M) )
        except Exception as ex:
            print(ex)
        self.dss = ConstSharedArray(data[0])
        self.OdotO = ConstSharedArray(data[1])
        self.Rdata = ConstSharedArray(np.exp(np.linspace(-25, 0, R)))



    def FDistributionPathname(self):
        return os.path.join(CorrFuncDir(self.M), "Fdata." + str(self.T) + ".np")

    def CorrDataPathname(self):
        return os.path.join(CorrFuncDir(self.M), "CorrData." + str(self.T) + ".np")


    def FDistribution(self):
        M = self.M
        T = self.T
        MakeDir(CorrFuncDir(M))
        res = []
        with fut.ProcessPoolExecutor() as exec:
            step = max(1, int(T/(mp.cpu_count()-1))+1)
            params = [[k,k+ step] for k in range(0,T, step)]
            params[-1][1] = T
            res = list(exec.map(self.SampleCorr, params))
        data = np.append(np.stack(res[:-1]),np.array(res[-1]))
        data.tofile(self.FDistributionPathname())

    def SampleCorr(self,params):
        beg,end = params
        M = self.M
        f = np.random.random()
        pq = Fraction(f).limit_denominator(int(M/2))
        p = pq.numerator
        q = pq.denominator
        beta = 2 * pi * p / q
        ar = []
        r = np.random.randint(0,int(M/q))
        if (M - q*r)%2 != 0:
            r +=1
        n = (M - q* r)//2
        for t in range(beg,end):
            alphas = np.array([beta] * (M - n) + [-beta] * n, dtype=float)
            np.random.shuffle(alphas)
            alphas = np.cumsum(alphas)
            alphas = np.append(alphas, alphas[0])

            FF = np.vstack([F(k,alphas, beta) for k in range(M)])
            FS = np.cumsum(FF, axis=0)
            Stot = FS[-1]
            m = np.random.randint(1,M)
            n = np.random.randint(0,m)
            Snm = FS[m] - FS[n]
            Smn = Stot - Snm
            snm = Snm / (m - n)
            smn = Smn / (n + M - m)
            ds = snm.real - smn.real
            ar.extend([beta,sqrt(ds.dot(ds)),np.dot(Omega(n,alphas,beta),Omega(m,alphas,beta)).real])
        return ar

    def GetCorr(self,params):
        beg,end = params
        s = self.dss[:,np.newaxis]* self.Rdata[np.newaxis,beg:end]
        s = sin(s)/s
        return self.OdotO[:].dot(s)/len(self.OdotO)

    def MakePlots(self):
        res = []
        R = self.R
        M = self.M
        T = self.T
        with fut.ProcessPoolExecutor() as exec:
            step = max(1, int(R/(mp.cpu_count()-1))+1)
            ranges = [[k,k+ step] for k in range(0,R, step)]
            ranges[-1][1] = R
            res = list(exec.map(self.GetCorr, ranges))
        corrdata =np.append(np.stack(res[:-1]),res[-1])
        corrdata.tofile(self.CorrDataPathname())
        print("made parallel map corrdata " + str(M) )
        plotpath = os.path.join(CorrFuncDir(M), "VortCorr.png")
        ok = corrdata>0
        XYPlot([self.Rdata[ok],corrdata[ok]],plotpath,logx=True,logy=True,title="vorticity-corr(r)")
        print("made VortCorr " + str(M) )



def test_FDistribution():
    M = 100001
    T = 10000
    R = 100000
    fdp = FDPlotter(M ,T ,R )
    fdp.MakePlots()

if __name__ == '__main__':
    test_FDistribution()
