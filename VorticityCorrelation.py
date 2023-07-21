import os

# from mpmath import rational
# here is a new line
from SortedArrayIter import SortedArrayIter
from Timer import MTimer as Timer
from plot import MakeDir, MakeNewDir, XYPlot, SubSampleWithErr, PlotXYErrBars, MultiPlot, MultiRankHistPos, MultiXYPlot
# from functools import reduce
# from operator import add

from numpy import pi, sin, cos, tan, sqrt, exp, log
import numpy as np
from parallel import ConstSharedArray, print_debug
import multiprocessing as mp
import concurrent.futures as fut
from fractions import Fraction


def CorrFuncDir(M):
    return os.path.join("plots", "VorticityCorr." + str(M))


def F(alphas, beta):
    return 1 / (2 * sin(beta / 2)) * np.array([cos(alphas), sin(alphas), 1j * cos(beta / 2) * np.ones_like(alphas)],
                                              complex)


def Omega(k, alphas, sigmas, beta):
    phi = alphas[k] + sigmas[k] * (beta / 2)
    return sigmas[k] / (2 * tan(beta / 2)) * np.array([cos(phi), sin(phi), 1j], complex)


class RandomFractions():
    @staticmethod
    def Pair(M):
        while (True):
            f = np.clip(np.random.random(), 0.5 / M, 1 - 0.5 / M)
            pq = Fraction(f).limit_denominator(M)
            if pq.denominator % 2 == M % 2:
                break
        return [pq.numerator, pq.denominator]

    def __init__(self, M, T):
        self.M = M
        self.T = T
        MakeDir(self.SaveDir())

    def SaveDir(self):
        return os.path.join("plots", "RandomFractions")

    def GetPathname(self):
        return os.path.join(self.SaveDir(), str(self.M) + ".np")

    def MakePairs(self):
        T = self.T
        M = self.M
        if os.path.isfile(self.GetPathname()):
            pairs = np.fromfile(self.GetPathname(), dtype=int).reshape(-1, 2)
            if len(pairs) >= T:
                return pairs[:T]
            pass
        res = []
        with fut.ProcessPoolExecutor() as exec:
            res = list(exec.map(self.Pair, [M] * T))
        pairs = np.array(res, dtype=int)
        pairs = np.unique(pairs, axis=0)
        np.random.shuffle(pairs)
        pairs = pairs[:self.T]
        pairs.tofile(self.GetPathname())
        return pairs


class CurveSimulator():
    def __init__(self, M, T, CPU, C):
        T_param = T
        MakeDir(CorrFuncDir(M))
        self.M = M
        self.CPU = CPU
        self.C = C
        self.Tstep = int(T / (4. * self.CPU)) + 1  # 4 to make last #CPU jobs equal. (+12% speed)
        T = self.Tstep * (T // self.Tstep)
        print(f"Adjusted parameter T: {T_param} --> {T}")
        self.T = T

    def GetSamples(self, params):
        beg, end = params
        ar = np.zeros((end - beg) * 3, dtype=float).reshape(-1, 3)
        np.random.seed(self.C + 1000 * beg)  # to make a unique seed for at least 1000 nodes
        i0 = self.Mindex(beg)
        M = (i0+1)*self.M
        sigmas = np.ones(M, dtype=int)  # +30% speed
        alphas = np.zeros(M,dtype=float)
        for k in range(beg, end):
            i = self.Mindex(k)
            if i > i0:
                i0 = i
                M = (i+1)*self.M
                sigmas = np.ones(M, dtype=int)  # +30% speed
                alphas = np.zeros(M,dtype=float)
            else:
                sigmas.fill(1)
                alphas.fill(0)
            p, q = RandomFractions.Pair(M)
            beta = (2 * pi * p) / float(q)
            N1, N2 = (M + q) // 2, (M - q) // 2
            if np.random.randint(2) == 1:
                N1, N2 = N2, N1
            sigmas[:N1].fill(-1)
            np.random.shuffle(sigmas)
            alphas[:] = np.cumsum(sigmas).astype(float) * beta
            m = np.random.randint(1, M)
            n = np.random.randint(0, m)
            Snm = np.sum(F(alphas[n:m], beta), axis=1)
            Smn = np.sum(F(alphas[m:M], beta), axis=1) + np.sum(F(alphas[0:n], beta), axis=1)
            snm = Snm / (m - n)
            smn = Smn / (n + M - m)
            ds = snm.real - smn.real
            t = k - beg
            ar[t, 0] = beta
            ar[t, 1] = sqrt(ds.dot(ds))
            ar[t, 2] = np.dot(Omega(n, alphas, sigmas, beta), Omega(m, alphas, sigmas, beta)).real
        return ar

    def PlotWithErr(self, xdata, ydata, name, num_samples=100):
        x, y, yerr = SubSampleWithErr(xdata, ydata, num_samples)
        plotpath = os.path.join(CorrFuncDir(self.M), name + ".png")
        PlotXYErrBars(x, y, yerr, plotpath, title=name)

    def FDistributionPathname(self):
        return os.path.join(CorrFuncDir(self.M), "Fdata." + str(self.T) + "." + str(self.C) + ".np")

    def Mindex(self, k):
        return (4 * k) // self.T

    def FDistribution(self):
        M = self.M
        T = self.T
        Tstep = self.Tstep
        MakeDir(CorrFuncDir(M))
        if not os.path.isfile(self.FDistributionPathname()):
            res = []
            params = [[k, k + Tstep] for k in range(0, T, Tstep)]
            with fut.ProcessPoolExecutor(max_workers=self.CPU - 1) as exec:
                res = list(exec.map(self.GetSamples, params))
            data = np.vstack(res)
            MakeNewDir(CorrFuncDir(M))
            data.tofile(self.FDistributionPathname())
        print("made FDistribution " + str(M))

    def ReadStatsFile(self, pathname):
        return np.fromfile(pathname, float).reshape(self.T, 3)

    def GetAllStats(self):
        pathnames = []
        for filename in os.listdir(CorrFuncDir(self.M)):
            if filename.endswith(".np") and filename.startswith("Fdata." + str(self.T) + "."):
                try:
                    splits = filename.split(".")
                    node_num = int(splits[-2])
                    if (node_num >= 0):
                        pathnames.append(os.path.join(CorrFuncDir(self.M), filename))
                except Exception as ex:
                    print_debug(ex)
                pass
            pass
        res = []
        with fut.ProcessPoolExecutor(max_workers=self.CPU - 1) as exec:
            res = list(exec.map(self.ReadStatsFile, pathnames))
        data = np.vstack(res).reshape(-1, self.T, 3)
        data.tofile(os.path.join(CorrFuncDir(self.M), 'AllStats.np'))

    def MakePlots(self):
        T = self.T
        M = self.M
        if not os.path.isfile(os.path.join(CorrFuncDir(self.M), 'AllStats.np')):
            self.GetAllStats()
        stats = np.fromfile(os.path.join(CorrFuncDir(self.M), 'AllStats.np'), float).reshape(-1, self.T, 3)
        stats = np.transpose(stats, (2, 1, 0))
        Betas = stats[0]
        Dss = stats[1]
        OdotO = stats[2]
        for x, name in zip([Betas, Dss, OdotO], ["beta", "DS", "OmOm"]):
            data = []
            for beg, end, m in [(T * k // 4, T * (k + 1) // 4, (k + 1) * M) for k in range(4)]:
                data.append([str(m), x[beg:end].reshape(-1)])
            plotpath = os.path.join(CorrFuncDir(M), "multi " + name + ".png")
            try:
                logx = (name in ("DS", "OmOm"))
                logy = (name != "beta")
                MultiRankHistPos(data, plotpath, name, logx=logx, logy=logy, num_subsamples=1000)
            except Exception as ex:
                print(ex)
        pass
        data = []
        for i, j, m in [(T * k // 4, T * (k + 1) // 4, (k + 1) * M) for k in range(4)]:
            oto = OdotO[i:j].reshape(-1)

            dss = Dss[i:j].reshape(-1)
            pos = oto > 0
            neg = oto < 0
            data.append([str(m), dss[pos], oto[pos]])
            data.append([str(-m), dss[neg], -oto[neg]])
        try:
            plotpath = os.path.join(CorrFuncDir(M), "multi OtOvsDss.png")
            MultiXYPlot(data, plotpath, logx=True, logy=True, title='OtoOVsDss', scatter=False, xlabel='log(dss)',
                        ylabel='log(oto)', frac_last=0.95, num_subsamples=1000)
        except Exception as ex:
            print(ex)
        print("made OtOvsDss " + str(M))

def test_FDistribution(M = 100000, T = 20000, CPU = mp.cpu_count(), C =0):
    with Timer("done FDistribution for M,T,C= " + str(M) + "," + str(T)+ "," + str(C)):
        fdp = CurveSimulator(M, T, CPU, C)
        fdp.FDistribution()# runs on each node, outputs placed in the plot dir of the main node

def MakePlots(M=100000, T=20000, CPU=mp.cpu_count()):
    with Timer("done MakePlots for M,T= " + str(M) + "," + str(T)):
        fdp = CurveSimulator(M, T, CPU, 0)
        fdp.MakePlots()  # runs on main node, pools tata if not yet done so,  subsamples data and makes plots


if __name__ == '__main__':
    import argparse
    import logging
    import multiprocessing as mp

    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', type=int, default=100003)
    parser.add_argument('-T', type=int, default=10000)
    parser.add_argument('-CPU', type=int, default=mp.cpu_count())
    parser.add_argument('-C', type=int, default=0)
    parser.add_argument('-debug', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-plot', action=argparse.BooleanOptionalAction, default=False)
    A = parser.parse_args()
    if A.C > 0:
        with Timer("done FDistribution for M,T= " + str(A.M) + "," + str(A.T)):
            test_FDistribution(A.M, A.T, A.CPU, A.C)
    else:
        MakePlots(A.M, A.T, A.CPU)

