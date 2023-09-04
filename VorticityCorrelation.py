import os, sys
from math import gcd

# from mpmath import rational
# here is a new line
from SortedArrayIter import SortedArrayIter
from Timer import MTimer as Timer
from plot import MakeDir, MakeNewDir, XYPlot, SubSampleWithErr, PlotXYErrBars, MultiPlot, MultiRankHistPos, MultiXYPlot, \
    RankHistPos, RankHist2
# from functools import reduce
# from operator import add
from RationalNumberGenerator import RationalRandom
from numpy import pi, sin, cos, tan, sqrt, exp, log
from numpy.linalg import multi_dot as mdot
import numpy as np
import multiprocessing as mp
import concurrent.futures as fut
from cfractions import Fraction
#from numba import vectorize, guvectorize, float64, int64
from QuadPy import SphericalFourierIntegral
from memory_profiler import profile
from scipy.special import betaln
# import jax.numpy as jnp
# from jax import jit
# from jax import random as jrandom
# import jax
# jax.config.update('jax_platform_name', 'cpu')

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
    return np.complex(np.cos(sigma * beta), np.sin(sigma * beta))



def DS_CPP(n, m, N_pos, N_neg, beta):
    INT64 = ctypes.c_int64
    libDS.DS.argtypes = (INT64, INT64, INT64, INT64, ctypes.c_double, c_double_p)
    libDS.DS.restype = ctypes.c_double
    np_o_o = np.zeros(1, dtype=float)
    dsabs = libDS.DS(n, m, N_pos, N_neg, beta, np_o_o.ctypes.data_as(c_double_p))
    return dsabs, np_o_o[0]

# void Corr( std::int64_t n, std::int64_t m, std::int64_t N_pos, std::int64_t N_neg, std::int64_t N_cor, double beta, /*IN*/ double* rho ,/*OUT*/ double* cor ) 
# def CORR_CPP(n, m, N_pos, N_neg, beta, rho_data):
#     INT64 = ctypes.c_int64
#     libDS.DS.argtypes = (INT64, INT64, INT64, INT64, ctypes.c_double, c_double_p)
#     libDS.DS.restype = ctypes.c_double
#     np_o_o = np.zeros(1, dtype=float)
#     dsabs = libDS.DS(n, m, N_pos, N_neg, beta, np_o_o.ctypes.data_as(c_double_p))
#     ans = np.ones_like(rho_data)
#     ans[1:] = sin(rho_data[1:]* dsabs)/(rho_data[1:]* dsabs)
#     return np_o_o[0] * ans


class RandomFractions():
    def Pairs(self,params):
        n,f0,f1= params
        M = self.mu
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
    def __init__(self, M, T, EG):
        self.mu = M
        self.T = T
        self.CPU = mp.cpu_count()
        self.EG = EG
        MakeDir(self.SaveDir())

    def SaveDir(self):
        return os.path.join("plots", "RandomFractions")

    def GetPairsPathname(self):
        return os.path.join(self.SaveDir(),"Pairs."+ str(self.mu) + ".np")
    def GetEnstrophyPathname(self):
        return os.path.join(self.SaveDir(),"Enstrophy."+ str(self.mu) + ".np")
    def MakePairs(self, f0, f1):
        if os.path.isfile(self.GetPairsPathname()):
            print(" Pairs exist in " + self.GetPairsPathname())
            return
        T = self.T
        M = self.mu
        res = []
        params = [((T * (i + 1) )// self.CPU - (T * i )// self.CPU, f0, f1) for i in range(self.CPU)]
        with fut.ProcessPoolExecutor(max_workers=self.CPU - 1) as exec:
            res = list(exec.map(self.Pairs, params))
        data = np.vstack(res)
        data.tofile(self.GetPairsPathname())
    def MakeEnstrophy(self):
        T = self.T
        M = self.mu
        res = []
        params = [((T * (i + 1)) // self.CPU - (T * i) // self.CPU) for i in range(self.CPU)]
        with fut.ProcessPoolExecutor(max_workers=self.CPU - 1) as exec:
            res = list(exec.map(self.mueanEnstrophy, params))
        data = np.vstack(res)
        data.tofile(self.GetEnstrophyPathname())
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
    M = 2_000_000_0003
    T = 1_000_000_00
    RF = RandomFractions(M,T)
    # RF.MakeEnstrophy()
    f0 = 0.5/M
    f1 = 1.-f0
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
    def __init__(self, M, EG, T, CPU, R0, R1, STP, C ):
        MakeDir(CorrFuncDir(M))
        self.mu = M
        self.CPU = CPU
        self.C = C
        self.T = T
        self.EG = EG
        self.R0 = R0
        self.R1 = R1
        self.STP = STP

    def GaussPair(self):
        M = self.mu
        while (True):
            f = np.clip(np.random.random(),1./M,1.-1./ M)
            pq = Fraction(f).limit_denominator(M)
            if pq.numerator == 0: continue
            if pq.denominator % 2 == M % 2:
                break
        return [pq.numerator, pq.denominator]

    def EulerPair(self):
        M = int(np.floor(0.5*np.random.exponential(1./self.muu)))*2
        while (True):
            q = np.random.randint(3,M)
            if q % 2 == M % 2:
                p = np.random.randint(1, q)
                if gcd(p,q) ==1:  # P( (p,q)=1 ) = phi(q)/q
                    if np.random.randint(0,M) < q:  # P( break ) = q/M * phi(q)/q = phi(q)/M ~ phi(q)
                        break
                    pass
                pass
            pass
        return [M, p, q]

        
    def GetSamples(self, params):
        beg, end = params
        ar = np.zeros((end - beg) * 3, dtype=float).reshape(-1, 3)
        np.random.seed(self.C + 1000 * beg)  # to make a unique seed for at least 1000 nodes

        for k in range(beg, end):
            M, p, q = self.EulerPair()
            beta = (2 * pi * p) / float(q)
            r =0
            N_pos = (M + q*r) // 2  # Number of 1's
            N_neg = M - N_pos  # Number of -1's

            n = np.random.randint(0, M)
            m = np.random.randint(n + 1, M + n) % M
            if n > m:
                n, m = m, n

            t = k - beg
            ar[t, 0] = 1/tan(beta/2)**2
            ar[t, 1], ar[t, 2] = DS_CPP(n, m, N_pos, N_neg, beta)
        return ar
    
    def FDistributionPathname(self):
        return os.path.join(CorrFuncDir(self.mu), "Fdata." + str(self.EG)+ "."+ str(self.T) + "." + str(self.C) + ".np")
    
    def FDistribution(self, serial):
        """
        :param serial: Boolean If set, run jobs serially.
        """
        mu = self.mu
        T = self.T
        MakeDir(CorrFuncDir(mu))
        if not os.path.isfile(self.FDistributionPathname()):
            res = None
            params = [(T * i // self.CPU, T * (i + 1) // self.CPU) for i in range(self.CPU)]
            if serial:
                res = list(map(self.GetSamples, params))
            else:
                with fut.ProcessPoolExecutor(max_workers=self.CPU) as exec:
                    res = list(exec.map(self.GetSamples, params))
            data = np.vstack(res)
            data.tofile(self.FDistributionPathname())
        print("made FDistribution " + str(mu))

    @staticmethod
    def ReadStatsFile(params):
        pathname, T , dim= params
        return np.fromfile(pathname, float).reshape(T, dim)

    
    def GetFDStats(self,mu):
        params= []
        T = None
        for filename in os.listdir(CorrFuncDir(mu)):
            if filename.endswith(".np") and filename.startswith("Fdata." + str(self.EG)):
                try:
                    splits = filename.split(".")
                    T = int(splits[-3])
                    if T > 0:
                        params.append([os.path.join(CorrFuncDir(mu), filename), T,3])
                except Exception as ex:
                    pass
                pass
            pass
        res = []
        with fut.ProcessPoolExecutor(max_workers=mp.cpu_count() -1) as exec:
            res = list(exec.map(self.ReadStatsFile, params))
        if res ==[]:
            raise Exception("no stats to collect!!!!")
        data = np.vstack(res).reshape(-1, T, 3)
        pathname = os.path.join(CorrFuncDir(self.mu), 'FDStats.' + str(self.EG)+ "." + str(T) + '.np')
        data.tofile(pathname)
        return pathname, T
        
    def MakePlots(self, Mulist):
        Betas =[]
        Dss = []
        OdotO = []
        np.sort(Mulist)
        for mu in Mulist:
            pathname = None
            T = None
            for filename in os.listdir(CorrFuncDir(mu)):
                if filename.endswith(".np") and filename.startswith("FDStats." + str(self.EG)):
                    try:
                        splits = filename.split(".")
                        T = int(splits[-2])
                        if (T > 0): 
                            pathname = os.path.join(CorrFuncDir(self.mu), filename)
                            break
                    except Exception as ex:
                        break
                    pass
            if pathname is None:
                pathname, T = self.GetFDStats(mu)
            stats = np.fromfile(pathname).reshape(-1, T, 3)
            stats = np.transpose(stats, (2, 1, 0)).reshape(3,-1)
            Betas.append(stats[0])
            Dss.append(stats[1])
            OdotO.append(stats[2])
        pass
        MaxMuu = Mulist[-1]
        for X, name in zip([Betas, Dss, OdotO, OdotO], ["logTanbeta", "DS", "OmOm", "-OmOm"]):
            data = []
            for k, mu in enumerate(Mulist):
                data.append([str(mu), X[k]]) if name != "-OmOm" else data.append([str(mu), -X[k]])
            plotpath = os.path.join(CorrFuncDir(MaxMuu), str(self.EG)+ "."  + name + ".png")
            try:
                logx = True 
                logy = True
                MultiRankHistPos(data, plotpath, name, logx=logx, logy=logy, num_subsamples=1000)
            except Exception as ex:
                print(ex)
        pass
        data = []
        for k, mu in enumerate(Mulist):
            oto = OdotO[k]
            dss =Dss[k]
            pos = oto > 0
            neg = oto < 0
            data.append([str(mu), dss[pos], oto[pos]])
            data.append([str(-mu), dss[neg], -oto[neg]])
            try:
                plotpath = os.path.join(CorrFuncDir(MaxMuu), str(self.EG)+ ".OtOvsDss.png")
                MultiXYPlot(data, plotpath, logx=True, logy=True, title='OtoOVsDss', scatter=False, xlabel='log(dss)',
                            ylabel='log(oto)', frac_last=0.9, num_subsamples=1000)
            except Exception as ex:
                print(ex)
        print("plotted otovsds " + str(MaxMuu))
    
        if self.STP<=0: return
        data =[]
        rho_data = np.linspace(self.R0,self.R1,self.STP)
        for k, mu in enumerate(Mulist):
            oto = OdotO[k]
            dss =Dss[k]
            corr = np.zeros_like(rho_data)
            rdx = rho_data[1:,np.newaxis]* dss[np.newaxis,:]
            otX = oto[np.newaxis,:]
            corr[1:] = np.mean(otX * sin(rdx)/rdx,axis=1)
            corr[0] = np.mean(oto)
            data.append([str(m), rho_data, corr])
        try:
            plotpath = os.path.join(CorrFuncDir(MaxMu), str(self.EG)+ ".CorrFunction.png")
            MultiXYPlot(data, plotpath, logx=False, logy=False, title='CorrFunction', scatter=False, xlabel='rho',
                        ylabel='cor', frac_last=0.9, num_subsamples=1000)
        except Exception as ex:
            print(ex)
        print("plotted CorrFunction " + str(MaxMu))
            


def test_FDistribution(Mu, EG, T, CPU, C, serial):
    """
    :param serial: Boolean If set, run serially.
    """
    with Timer("done FDistribution for M,T,C= " + str(Mu) + "," + str(T)+ "," + str(C)):
        fdp = CurveSimulator(Mu, EG, T, CPU, 0, 0, 0, C)
        fdp.FDistribution(serial)  # runs on each node, outputs placed in the plot dir of the main node

    
def MakePlots(Mu, EG, T, CPU, R0, R1, STP):
    with Timer("done MakePlots for Mu,T= " + str(Mu) + "," + str(T)):
        fdp = CurveSimulator(Mu, EG, T, CPU, R0, R1, STP, 0)
        fdp.MakePlots([Mu])  # runs on main node, pools tata if not yet done so,  subsamples data and makes plots

#from euler_maths import euler_totients
#from primefac import factorint

# @jit
def euler_totients(N: int) -> list:
    """Returns list with phi(i) at index i for i < N."""
    phi = np.zeros(N, int)
    for i in range(2, N):
        if phi[i] == 0:
            phi[i] = i - 1
            for j in range(2 * i, N, i):
                if phi[j] == 0:
                    phi[j] = j
                phi[j] = (i - 1) * phi[j] // i
    return phi

class EulerPhi():
    def __init__(self):
        MakeDir(self.SaveDir())
        pass

    def SaveDir(self):
        return os.path.join("plots", "EulerPhi")

    def GetProbPathname(self, N):
        return os.path.join(self.SaveDir(),"Probs."+ str(N) + ".np")

    def Probability(self,N):
        phis = euler_totients(N)
        phis.tofile(self.GetProbPathname(N))

def test_EulerPhi():
    N = 10_000_000
    ep =EulerPhi()
    ep.Probability(N)

if __name__ == '__main__':
    # test_Numba()
    import argparse
    import logging
    import multiprocessing as mp

    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', type=int, default=10_000_001)
    parser.add_argument('-EG', type=str, default='E')
    parser.add_argument('-T', type=int, default=10000)
    parser.add_argument('-CPU', type=int, default=mp.cpu_count())
    parser.add_argument('-C', type=int, default=1)
    parser.add_argument('--serial', default=False, action="store_true")
    parser.add_argument('-R0', type=float, default=0.0)
    parser.add_argument('-R1', type=float, default=0.01)
    parser.add_argument('-STP', type=int, default=100000)
    
    A = parser.parse_args()
    if A.C > 0:
        with Timer("done Distribution for M,T= " + str(A.M) + "," + str(A.T)):
            test_FDistribution(A.M, A.EG, A.T, A.CPU, A.C, A.serial)

    MakePlots(A.M, A.EG, A.T, A.CPU, A.R0, A.R1, A.STP)
   

