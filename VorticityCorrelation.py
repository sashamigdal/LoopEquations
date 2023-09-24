import os, sys
from Timer import MTimer as Timer
from plot import MakeDir, XYPlot, RankHistPos, RankHist2
from numpy import sqrt
from numpy.linalg import multi_dot as mdot
import numpy as np
import multiprocessing as mp
import concurrent.futures as fut
from cfractions import Fraction
from QuadPy import SphericalFourierIntegral
from CurveSimulator import CurveSimulator


# @jit
def F(sigma, beta):
    return np.complex(np.cos(sigma * beta), np.sin(sigma * beta))


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





def test_FDistribution(Mu, EG, T, CPU, C, serial):
    """
    :param serial: Boolean If set, run serially.
    """
    if sys.platform == 'linux':
        with Timer("done FDistribution for M,T,C= " + str(Mu) + "," + str(T)+ "," + str(C)):
            fdp = CurveSimulator(Mu, EG, T, CPU, 0, 0, 0, C)
            fdp.FDistribution(serial)  # runs on each node, outputs placed in the plot dir of the main node
    else:
        print(f"not implemented on {sys.platform}")

def test_Spectrum(Mu, EG, T, CPU, C, serial, Nlam, gamma):
    """
    :param serial: Boolean If set, run serially.
    """
    if sys.platform == 'linux':
        with Timer("done Spectrum for M,T,C= " + str(Mu) + "," + str(T)+ "," + str(C)):
            fdp = CurveSimulator(Mu, EG, T, CPU, 0, 0, 0, C, Nlam, gamma)
            fdp.PrepareSpectrum(serial)  # runs on each node, outputs placed in the plot dir of the main node
    else:
        print(f"not implemented on {sys.platform}")


def MakePlots(Mu, EG, T, CPU, R0, R1, STP):
    with Timer("done MakePlots for Mu,T= " + str(Mu) + "," + str(T)):
        fdp = CurveSimulator(Mu, EG, T, CPU, R0, R1, STP, 0)
        fdp.MakePlots([Mu])  # runs on main node, pools tata if not yet done so,  subsamples data and makes plots

def test_makePlots(Mu=1e-7, EG="E", T=1000, CPU=0, R0=0.001, R1=0.003, STP=1000):
    MakePlots(Mu, EG, T, CPU, R0, R1, STP)
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
    parser.add_argument('-Mu', type=float, default=1e-3)
    parser.add_argument('-EG', type=str, default='E')
    parser.add_argument('-T', type=int, default=1000)
    parser.add_argument('-CPU', type=int, default=mp.cpu_count())
    parser.add_argument('-C', type=int, default=1)
    parser.add_argument('--serial', default=False, action="store_true")
    parser.add_argument('-R0', type=float, default=0.001)
    parser.add_argument('-R1', type=float, default=0.0035)
    parser.add_argument('-STP', type=int, default=10000)
    parser.add_argument('-NLAM', type=int, default=1)
    parser.add_argument('-GAMMA', type=float, default=-100.)
    #, Nlam, gamma
    
    A = parser.parse_args()
    if A.C > 0:
        if A.NLAM >0:
            with Timer("done Spectrum for Mu,T= "   + str(A.Mu) + "," + str(A.T)+"," +  str(A.NLAM) + "," + str(A.GAMMA)):
                test_Spectrum(A.Mu, A.EG, A.T, A.CPU, A.C, A.serial, A.NLAM, A.GAMMA)
        else:
            with Timer("done Distribution for Mu,T= " + str(A.Mu) + "," + str(A.T)):
                test_FDistribution(A.Mu, A.EG, A.T, A.CPU, A.C, A.serial)
    else:
        MakePlots(A.Mu, A.EG, A.T, A.CPU, A.R0, A.R1, A.STP)
   

