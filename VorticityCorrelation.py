import os

# from mpmath import rational
# here is a new line
from SortedArrayIter import SortedArrayIter
from Timer import MTimer as Timer
from plot import MakeDir, MakeNewDir, XYPlot, SubSampleWithErr, PlotXYErrBars, MultiPlot, MultiRankHistPos, MultiXYPlot
# from functools import reduce
# from operator import add

from numpy import pi, sin, cos, tan, sqrt, exp, log
from numpy.linalg import multi_dot as mdot
import numpy as np
import multiprocessing as mp
import concurrent.futures as fut
from fractions import Fraction
#from numba import vectorize, guvectorize, float64, int64
from QuadPy import SphericalFourierIntegral
from memory_profiler import profile

def CorrFuncDir(M):
    return os.path.join("plots", "VorticityCorr." + str(M))


def F(sigma, beta):
    return 1 / (2 * sin(beta / 2)) * np.array([cos(sigma * beta), sin(sigma * beta)],dtype= float)


def Omega(k, sigmas, beta):
    M = len(sigmas)
    k1 = (k+1)%M
    phi = (sigmas[k1] + sigmas[k]) * (beta / 2)
    return (sigmas[k1]- sigmas[k]) / (2 * tan(beta / 2)) * np.array([cos(phi), sin(phi), 1j], complex)

# @profile
def SS(n, m, M, sigmas, beta):
    Snm = np.sum(F(sigmas[n:m], beta), axis=1)
    Smn = np.sum(F(sigmas[m:M], beta), axis=1) + np.sum(F(sigmas[0:n], beta), axis=1)
    snm = Snm / (m - n)
    smn = Smn / (n + M - m)
    ds = snm - smn
    return np.sqrt(ds.dot(ds))


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


class GroupFourierIntegral:
    def __init__(self):
        s0 = np.matrix([[1, 0], [0, 1]])
        s1 = np.matrix([[0, 1], [1, 0]])
        s2 = np.matrix([[0, -1j], [1j, 0]])
        s3 = np.matrix([[1, 0], [0, -1]])
        self.Sigma = [s1, s2, s3]  # Pauli matrices
        self.Tau = [s0, 1j * s1, 1j * s2, 1j * s3]  # quaternions
        TT =np.array(
                        [np.trace(mdot([self.Sigma[i], self.Tau[al], self.Sigma[j], self.Tau[be].H]))
                                        for i in range(3)
                                    for j in range(3)
                                for al in range(4)
                            for be in range(4)
                        ]
                    ).reshape((3,3,4,4))
        TT +=  np.transpose(TT,(0,1,3,2))
        self.TT = np.transpose(TT * 0.5,(2,3,0,1))
    def GetRMatrix(self, X):
        # using quaternionic representation for tr_3 (O(3).X)
        # O(3)_{i j} = q_a q_b tr_2 (s_i.Tau_a.s_j.Tau^H_b)
        # Q = q_a Tau_a = q.Tau
        # O(3)_{i j} = tr_2 (s_i.Q.s_j.Q.H)
        #  V1.O(3).V2 =  tr_3( O(3).kron(V2,V1)) = O(3)_{i j} V2_j V1_i = tr_2 (HV1.Q.Hv2.Q.H)
        Y = np.dot(X,self.TT)
        return np.trace(Y,axis1=0, axis2=3)

    def Integral(self, t):
        return SphericalFourierIntegral(self.rr / np.sqrt(t))
    def FourierIntegralQuadpy(self, X, t0, t1, time_steps):
        self.rr = self.GetRMatrix(X)

        res = []
        with fut.ProcessPoolExecutor() as exec:
            res = list(exec.map(self.Integral, np.linspace(t0, t1, time_steps)))
        return res

    def __del__(self):
        pass


def test_GroupFourierIntegral():
    gfi = GroupFourierIntegral()
    r = np.random.normal(scale=0.1, size=30) + 1j * np.random.normal(scale=0.1, size=30)
    r = r.reshape((3, 10))
    X = r.dot(r.T)
    with Timer("multi integrals in QuadPy"):
        test1 = gfi.FourierIntegralQuadpy(X,  1, 2, 1000)
    # print(test1)
    pass


class CurveSimulator():
    def __init__(self, M, T, CPU, C):
        MakeDir(CorrFuncDir(M))
        self.M = M
        self.CPU = CPU
        self.C = C
        self.T = T
    
    def GetSamples(self, params):
        beg, end = params
        ar = np.zeros((end - beg) * 3, dtype=float).reshape(-1, 3)
        np.random.seed(self.C + 1000 * beg)  # to make a unique seed for at least 1000 nodes
        M = self.M
        sigmas = np.ones(M, dtype=int)  # +30% speed
        for k in range(beg, end):
            sigmas.fill(1)
            p, q = RandomFractions.Pair(M)
            beta = (2 * pi * p) / float(q)
            N1, N2 = (M + q) // 2, (M - q) // 2
            if np.random.randint(2) == 1:
                N1, N2 = N2, N1
            m = np.random.randint(1, M)
            n = np.random.randint(0, m)
            #########to be parallemized on GPU
            sigmas[:N1].fill(-1)
            np.random.shuffle(sigmas)
            sigmas[:] = np.cumsum(sigmas)
            dsabs = SS(n,m,M, sigmas, beta)
            #################################################
            t = k - beg
            ar[t, 0] = beta
            ar[t, 1] = dsabs
            ar[t, 2] = np.dot(Omega(n, sigmas, beta), Omega(m,sigmas, beta)).real
        return ar


    def FDistributionPathname(self):
        return os.path.join(CorrFuncDir(self.M), "Fdata." + str(self.T) + "." + str(self.C) + ".np")

    
    def FDistribution(self):
        M = self.M
        T = self.T
        MakeDir(CorrFuncDir(M))
        if not os.path.isfile(self.FDistributionPathname()):
            res = []
            params = [(T * i // self.CPU, T * (i + 1) // self.CPU) for i in range(self.CPU)]
            # with fut.ProcessPoolExecutor(max_workers=self.CPU - 1) as exec:
            res = list(map(self.GetSamples, params))
            data = np.vstack(res)
            data.tofile(self.FDistributionPathname())
        print("made FDistribution " + str(M))

    def ReadStatsFile(self, pathdata):
        pathname, T = pathdata
        return np.fromfile(pathname, float).reshape(T, 3)

    
    def GetAllStats(self,M):
        pathdata= []
        T = None
        for filename in os.listdir(CorrFuncDir(M)):
            if filename.endswith(".np") and filename.startswith("Fdata."):
                try:
                    splits = filename.split(".")
                    T = int(splits[-3])
                    node_num = int(splits[-2])
                    if (node_num >= 0 and T >0):
                        pathdata.append([os.path.join(CorrFuncDir(M), filename), T])
                except Exception as ex:
                    print_debug(ex)
                pass
            pass
        res = []
        with fut.ProcessPoolExecutor(max_workers=mp.cpu_count() -1) as exec:
            res = list(exec.map(self.ReadStatsFile, pathdata))
        if res ==[]:
            raise Exception("no stats to collect!!!!")
        data = np.vstack(res).reshape(-1, T, 3)
        pathname = os.path.join(CorrFuncDir(self.M), 'AllStats.' + str(T) + '.np')
        data.tofile(pathname)
        return pathname
        
    def MakePlots(self, Mlist):
        Betas =[]
        Dss = []
        OdotO = []
        np.sort(Mlist)
        for M in Mlist:
            pathname = None
            T = None
            for filename in os.listdir(CorrFuncDir(M)):
                if filename.endswith(".np") and filename.startswith("AllStats."):
                    try:
                        splits = filename.split(".")
                        T = int(splits[-2])
                        if (T > 0): 
                            pathname = os.path.join(CorrFuncDir(self.M), filename)
                            break
                    except Exception as ex:
                        break
                    pass
            if pathname is None:
                pathname = self.GetAllStats(M)
            stats = np.fromfile(pathname).reshape(-1, T, 3)
            stats = np.transpose(stats, (2, 1, 0)).reshape(3,-1)
            Betas.append(stats[0])
            Dss.append(stats[1])
            OdotO.append(stats[2])
        pass
        MaxM = Mlist[-1]
        for X, name in zip([Betas, Dss, OdotO], ["beta", "DS", "OmOm"]):
            data = []
            for k, m in enumerate(Mlist):
                data.append([str(m), X[k]])
            plotpath = os.path.join(CorrFuncDir(MaxM), "multi " + name + ".png")
            try:
                logx = (name in ("DS", "OmOm"))
                logy = (name != "beta")
                MultiRankHistPos(data, plotpath, name, logx=logx, logy=logy, num_subsamples=1000)
            except Exception as ex:
                print(ex)
        pass
        data = []
        for k, m in enumerate(Mlist):
            oto = OdotO[k]
            dss =Dss[k]
            pos = oto > 0
            neg = oto < 0
            data.append([str(m), dss[pos], oto[pos]])
            data.append([str(-m), dss[neg], -oto[neg]])
        try:
            plotpath = os.path.join(CorrFuncDir(MaxM), "multi OtOvsDss.png")
            MultiXYPlot(data, plotpath, logx=True, logy=True, title='OtoOVsDss', scatter=False, xlabel='log(dss)',
                        ylabel='log(oto)', frac_last=0.95, num_subsamples=1000)
        except Exception as ex:
            print(ex)
        print("made OtOvsDss " + str(MaxM))

def test_FDistribution(M = 100000, T = 20000, CPU = mp.cpu_count(), C =0):
    with Timer("done FDistribution for M,T,C= " + str(M) + "," + str(T)+ "," + str(C)):
        fdp = CurveSimulator(M, T, CPU, C)
        fdp.FDistribution()# runs on each node, outputs placed in the plot dir of the main node

def MakePlots(M=100000, T=20000, CPU=mp.cpu_count()):
    with Timer("done MakePlots for M,T= " + str(M) + "," + str(T)):
        fdp = CurveSimulator(M, T, CPU, 0)
        fdp.MakePlots([M])  # runs on main node, pools tata if not yet done so,  subsamples data and makes plots

def test_Numba():
    M = 1000
    sigmas = np.ones(M,dtype=int)
    alphas = np.zeros(M, dtype=float)
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
    Snm, Smn = SS(n, m, M, alphas, beta)
    return Snm, Smn


if __name__ == '__main__':
    # test_Numba()
    import argparse
    import logging
    import multiprocessing as mp

    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', type=int, default=50000)
    parser.add_argument('-T', type=int, default=1000)
    parser.add_argument('-CPU', type=int, default=mp.cpu_count())
    parser.add_argument('-C', type=int, default=1)
    A = parser.parse_args()
    if A.C > 0:
        with Timer("done FDistribution for M,T= " + str(A.M) + "," + str(A.T)):
            test_FDistribution(A.M, A.T, A.CPU, A.C)
    else:
        MakePlots(A.M, A.T, A.CPU)

