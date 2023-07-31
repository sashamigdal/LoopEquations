import os, sys

# from mpmath import rational
# here is a new line
from SortedArrayIter import SortedArrayIter
from Timer import MTimer as Timer
from plot import MakeDir, MakeNewDir, XYPlot, SubSampleWithErr, PlotXYErrBars, MultiPlot, MultiRankHistPos, MultiXYPlot, \
    RankHistPos, RankHist2
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

# import jax.numpy as jnp
# from jax import jit
# from jax import random as jrandom
# import jax
#jax.config.update('jax_platform_name', 'cpu')

import ctypes
libDS_path = os.path.join("CPP/cmake-build-release", 'libDS.so')
sys.path.append("CPP/cmake-build-release")
libDS = ctypes.cdll.LoadLibrary(libDS_path)
c_double_p = ctypes.POINTER(ctypes.c_double)
c_int64_p = ctypes.POINTER(ctypes.c_int64)
c_uint64_p = ctypes.POINTER(ctypes.c_uint64)
c_size_t_p = ctypes.POINTER(ctypes.c_size_t)
def CorrFuncDir(M):
    return os.path.join("plots", "VorticityCorr." + str(M))


# @jit
def F(sigma, beta):
    return jax.lax.complex(jax.lax.cos(sigma * beta), jax.lax.sin(sigma * beta))


# /-\frac{\sin (\beta  \sigma )}{2 (\cos (\beta )-1)}
def Omega(k, sigmas, beta):
    return sin(beta * sigmas[k]) / (2 * (1 - cos(beta)))


# @jit
def DS_Python(mask_nm, M, sigmas, beta, prng_key):
    sigmas = jax.random.permutation(prng_key,sigmas)
    sigmas =jnp.cumsum(sigmas, axis=0)
    mask_mn = 1 - mask_nm
    FF = F(sigmas, beta)
    Snm = jnp.sum(FF * mask_nm, axis=0)
    Smn = jnp.sum(FF * mask_mn, axis=0)
    len_nm = jnp.sum(mask_nm)
    snm = Snm / len_nm
    smn = Smn / (M - len_nm)
    ds = snm - smn
    return jnp.abs(ds/ (2 * jnp.sin(beta / 2)))

def DS_CPP(n, m, N_pos, N_neg, beta):
    INT64 = ctypes.c_int64
    libDS.DS.argtypes = (INT64, INT64, INT64, INT64, ctypes.c_double, c_double_p)
    libDS.DS.restype = ctypes.c_double
    np_o_o = np.zeros(1, dtype=float)
    dsabs = libDS.DS(n, m, N_pos, N_neg, beta, np_o_o.ctypes.data_as(c_double_p))
    return dsabs, np_o_o[0]


def DS(n, m, M, sigmas, beta):
    return DS_Python(n, m, M, sigmas, beta)
class RandomFractions():
    def Pairs(self,params):
        n,f0,f1= params
        M = self.M
        pairs = np.zeros((n, 2), dtype=int)
        l = 0
        while (l < n):
            f = np.random.uniform(low=f0, high=f1)
            pq = Fraction(f).limit_denominator(M)
            if pq.denominator % 2 == M % 2:
                pairs[l, 0] = pq.numerator
                pairs[l, 1] = pq.denominator
                l+=1
        return pairs
    def __init__(self, M, T):
        self.M = M
        self.T = T
        self.CPU = mp.cpu_count()
        MakeDir(self.SaveDir())

    def SaveDir(self):
        return os.path.join("plots", "RandomFractions")

    def GetPairsPathname(self):
        return os.path.join(self.SaveDir(),"Pairs."+ str(self.M) + ".np")

    def MakePairs(self, f0, f1):
        if os.path.isfile(self.GetPairsPathname()):
            print(" Pairs exist in " + self.GetPairsPathname())
            return
        T = self.T
        M = self.M
        res = []
        params = [((T * (i + 1) )// self.CPU - (T * i )// self.CPU, f0, f1) for i in range(self.CPU)]
        with fut.ProcessPoolExecutor(max_workers=self.CPU - 1) as exec:
            res = list(exec.map(self.Pairs, params))
        data = np.vstack(res)
        data.tofile(self.GetPairsPathname())

    def ReadPairsFile(self, params):
        pathname, M = params
        return np.fromfile(pathname,dtype = int).reshape(-1,2)

    def GetAllStatsPathname(self, M):
        return os.path.join(self.SaveDir(),"AllStats." + str(M) + ".np")


    def MakePlots(self):
        datas = dict()
        for filename in os.listdir(self.SaveDir()):
            if filename.endswith(".np") and filename.startswith("Pairs."):
                try:
                    splits = filename.split(".")
                    M = int(splits[-2])
                    if (M > 0):
                        array = np.fromfile(os.path.join(self.SaveDir(), filename), dtype=int).reshape(-1,2)
                        if M in datas:
                            datas[M] = np.append(datas[M], array)
                        else:
                            datas[M] = array
                except Exception as ex:
                    pass
                pass
            pass
        for M, data in datas.items():
            pairs = data.T
            pp = pairs[0].astype(float)
            qq = pairs[1].astype(float)
            eps = pp / qq - 0.5
            scale = sqrt(float(M))
            lim = float(M)
            ok = (qq < lim) & (eps > -0.5) & (eps < 0.5)
            xx = qq[ok]/scale
            yy = eps[ok] * scale
            RankHistPos(xx,os.path.join(self.SaveDir(),"qdata." + str(M) + ".png"),name='RankHist', var_name='\\eta',
                        logx=False, logy=False, max_tail=0.9, min_tail=0.05)
            RankHist2(yy, os.path.join(self.SaveDir(),"epsdata." + str(M) + ".png"), name='RankHist', var_name='\\eta', logx=False, logy=False)
            ord = np.argsort(xx)
            XYPlot([xx[ord],yy[ord]],os.path.join(self.SaveDir(),"eps_vs_q." + str(M) + ".png"), logx=False, logy=False)

def test_RandomFractions():
    M = 10_000_000_000
    T = 10_000_000
    gap= 10./sqrt(M)
    f0 = 0.5 - gap
    f1 = 0.5 + gap
    RF = RandomFractions(M,T)
    RF.MakePairs(f0,f1)
    # RF.MakePlots()

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

    def Pair(self):
        M = self.M
        while (True):
            f = np.clip(np.random.random(), 0.5 / M, 1 - 0.5 / M)
            pq = Fraction(f).limit_denominator(M)
            if pq.denominator % 2 == M % 2:
                break
        return [pq.numerator, pq.denominator]

    def GetSamples(self, params):
        beg, end = params
        ar = np.zeros((end - beg) * 3, dtype=float).reshape(-1, 3)
        np.random.seed(self.C + 1000 * beg)  # to make a unique seed for at least 1000 nodes
        # prng_key = jrandom.PRNGKey(self.C + 1000 * beg)
        M = self.M
        for k in range(beg, end):
            p, q = self.Pair()
            beta = (2 * pi * p) / float(q)

            N_pos = (M + q) // 2  # Number of 1's
            N_neg = (M - q) // 2  # Number of -1's
            if np.random.randint(2) == 1:
                N_pos, N_neg = N_neg, N_pos

            n = np.random.randint(0, M)
            m = np.random.randint(n+1, M+n) % M
            if n > m:
                n, m = m, n

            # mask_nm = jnp.zeros(M, dtype=int)
            # mask_nm = mask_nm.at[n:m].set(1)

            t = k - beg
            ar[t, 0] = beta
            ar[t, 1], ar[t, 2] = DS_CPP(n, m, N_pos, N_neg, beta)
        return ar


    def FDistributionPathname(self):
        return os.path.join(CorrFuncDir(self.M), "Fdata." + str(self.T) + "." + str(self.C) + ".np")

    
    def FDistribution(self, serial):
        """
        :param serial: Boolean If set, run jobs serially.
        """
        M = self.M
        T = self.T
        MakeDir(CorrFuncDir(M))
        if not os.path.isfile(self.FDistributionPathname()):
            res = None
            params = [(T * i // self.CPU, T * (i + 1) // self.CPU) for i in range(self.CPU)]
            if serial:
                res = list(map(self.GetSamples, params))
            else:
                with fut.ProcessPoolExecutor(max_workers=self.CPU - 1) as exec:
                    res = list(exec.map(self.GetSamples, params))
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
                    pass
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
        return pathname, T
        
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
                pathname, T = self.GetAllStats(M)
            stats = np.fromfile(pathname).reshape(-1, T, 3)
            stats = np.transpose(stats, (2, 1, 0)).reshape(3,-1)
            Betas.append(stats[0])
            Dss.append(stats[1])
            OdotO.append(stats[2])
        pass
        MaxM = Mlist[-1]
        for X, name in zip([Betas, Dss, OdotO, OdotO], ["beta", "DS", "OmOm", "-OmOm"]):
            data = []
            for k, m in enumerate(Mlist):
                data.append([str(m), X[k]]) if name != "-OmOm" else data.append([str(m), -X[k]])
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

def test_FDistribution(M, T, CPU, C, serial):
    """
    :param serial: Boolean If set, run serially.
    """
    with Timer("done FDistribution for M,T,C= " + str(M) + "," + str(T)+ "," + str(C)):
        fdp = CurveSimulator(M, T, CPU, C)
        fdp.FDistribution(serial)  # runs on each node, outputs placed in the plot dir of the main node


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
    Snm, Smn = DS(n, m, M, alphas, beta)
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
    parser.add_argument('--serial', default=False, action="store_true")
    A = parser.parse_args()
    if A.C > 0:
        with Timer("done FDistribution for M,T= " + str(A.M) + "," + str(A.T)):
            test_FDistribution(A.M, A.T, A.CPU, A.C, A.serial)
    else:
        MakePlots(A.M, A.T, A.CPU)

