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

from SortedArrayIter import SortedArrayIter
from Timer import MTimer as Timer
from plot import MakeDir, XYPlot, SubSampleWithErr, PlotXYErrBars, MultiPlot, MultiRankHistPos, MultiXYPlot
# from functools import reduce
# from operator import add

from numpy import pi, sin, cos, tan, sqrt, exp, log
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
    def GetSamples(self, params):
        beg, end = params
        ar = np.zeros((end-beg)*3,dtype=float).reshape(-1,3)
        for k in range(beg, end):
            M = self.MM(k,self.T)
            q = 2-(M%2) +2 *np.random.randint((M+1)//2)
            p = np.random.randint(1,q)
            beta = 2 * pi * p / q
            alphas = np.array([1] * ((M+q)//2) + [-1] * ((M-q)//2), dtype=int)
            np.random.shuffle(alphas)
            alphas = np.cumsum(alphas)
            alphas = np.append(alphas, alphas[0]).astype(float) * beta

            FF = np.vstack([F(k, alphas, beta) for k in range(M)])
            FS = np.cumsum(FF, axis=0)
            Stot = FS[-1]
            m = np.random.randint(1, M)
            n = np.random.randint(0, m)
            Snm = FS[m] - FS[n]
            Smn = Stot - Snm
            snm = Snm / (m - n)
            smn = Smn / (n + M - m)
            ds = snm.real - smn.real
            t = k-beg
            ar[t,0] = beta
            ar[t,1] = sqrt(ds.dot(ds))
            ar[t,2] = np.dot(Omega(n, alphas, beta), Omega(m, alphas, beta)).real
        return ar

    def GetCorr(self, params):
        i,j, beg, end  = params
        s = self.dss[i:j, np.newaxis] * self.Rdata[np.newaxis, beg:end]
        s = sin(s) / s
        return self.OdotO[i:j].dot(s)/(j-i)

    def PlotWithErr(self, xdata, ydata, name, num_samples=100):
        x, y, yerr =SubSampleWithErr(xdata,ydata, num_samples )
        plotpath = os.path.join(CorrFuncDir(self.M),name + ".png")
        PlotXYErrBars(x, y, yerr, plotpath,title=name)

    def FDistributionPathname(self):
        return os.path.join(CorrFuncDir(self.M), "Fdata." + str(self.T) + ".np")

    def CorrDataPathname(self):
        return os.path.join(CorrFuncDir(self.M), "CorrData." + str(self.T) + ".np")
    def MDataPathname(self):
        return os.path.join(CorrFuncDir(self.M), "MData." + str(self.T) + ".np")

    def MM(self, k, T):
        if k < T//4:  return  self.M
        if k < T//2:  return 2* self.M
        if k < (3 * T)//4: return  3 * self.M
        return  4 * self.M

    def FDistribution(self):
        M = self.M
        T = self.T
        R = self.R
        Tstep = self.Tstep

        MakeDir(CorrFuncDir(M))
        try:
            data = np.fromfile(self.FDistributionPathname(),dtype=float).reshape(-1,3)
        except:
            res = []
            params = [[k, k + Tstep ] for k in range(0, T, Tstep)]
            with fut.ProcessPoolExecutor() as exec:
                res = list(exec.map(self.GetSamples, params))
            data = np.vstack(res)
            data.tofile(self.FDistributionPathname())
            print("made FDistribution " + str(M))
        data = data.T
        self.betas = ConstSharedArray(data[0])
        self.dss = ConstSharedArray(data[1])
        self.OdotO = ConstSharedArray(data[2])
        self.Rdata = ConstSharedArray(np.linspace(1./M, 0.5, R))


    def MakeCorrData(self):
        M = self.M
        R = self.R
        T = self.T
        Rstep = self.Rstep
        if os.path.isfile(self.CorrDataPathname()):
            print("already made parallel map GetCorr " + str(M))
            return

        dd =[]
        for i,j, m in [(T*k//4,T*(k+1)//4,(k+1)*M) for k in range(4)]:
            res = []
            with fut.ProcessPoolExecutor() as exec:
                ranges = [[ i,j, k, k + Rstep] for k in range(0, R, Rstep)]
                res.extend(list(exec.map(self.GetCorr, ranges)))
            dd.extend([np.hstack(res)])
        corrdata = np.stack(dd)
        corrdata.tofile(self.CorrDataPathname())
        print("made parallel map GetCorr " + str(M))


    def __init__(self, M, T, R):
        MakeDir(CorrFuncDir(M))
        self.M = M
        self.Tstep = max(1, int(T / (mp.cpu_count() - 1)) + 1)
        T = self.Tstep * (T//self.Tstep)
        self.T = T
        self.Rstep = max(1, int(R / (mp.cpu_count() - 1)) + 1)
        R =  self.Rstep * (R//self.Rstep)
        self.R = R
        self.FDistribution()
        self.MakeCorrData()
        for x, name in zip([self.betas,self.dss,-self.OdotO[:], self.OdotO[:]],["beta","DS","-OmOm","OmOm"]):
            data = []
            for beg,end, m in [(T*k//4,T*(k+1)//4,(k+1)*M) for k in range(4)]:
                data.append([str(m),x[beg:end]])
            plotpath = os.path.join(CorrFuncDir(M),"multi " + name + ".png")
            try:
                MultiRankHistPos(data,plotpath,name,logx=True,logy=True)
            except Exception as ex:
                print(ex)
        pass
        corrdata = np.fromfile(self.CorrDataPathname(),dtype=float).reshape(4,-1)
        datapos = []
        dataneg= []
        for i, m in enumerate([self.M, self.M *2,self.M * 3, self.M *4 ]):
            pos = corrdata[i] >0
            neg = corrdata[i] <0
            datapos.append([str(m),self.Rdata[pos],corrdata[i][pos]])
            dataneg.append([str(m),self.Rdata[neg],-corrdata[i][neg]])
        plotpath = os.path.join(CorrFuncDir(M),"multicorrPos.png")
        try:
            MultiXYPlot(datapos, plotpath, logx=True, logy=True, title='VortCorrPos',scatter=False, xlabel ='r', ylabel='corr')
        except Exception as ex:
            print(ex)
        plotpath = os.path.join(CorrFuncDir(M),"multicorrNeg.png")
        try:
            MultiXYPlot(dataneg, plotpath, logx=True, logy=True, title='VortCorrNeg',scatter=False, xlabel ='r', ylabel='-corr')
        except Exception as ex:
            print(ex)
        print("made VortCor " + str(M))


def test_FDistribution():
    M = 1000000
    T = 1000
    R = 1000
    with Timer("done FDistribution for M,T,R= " + str(M) + "," + str(T) + "," + str(R)):
        fdp = FDPlotter(M, T, R)

if __name__ == '__main__':
    test_FDistribution()
