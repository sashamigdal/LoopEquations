from math import gcd

import scipy

import numpy as np
import os, sys
from numpy import cos, sin, pi, linspace

import mpmath as mpm

import sdeint
import multiprocessing as mp

from wolframclient.evaluation import WolframLanguageSession

from ComplexToReal import ComplexToRealVec, RealToComplexVec, ComplextoRealMat, ReformatRealMatrix, MaxAbsComplexArray, \
    SumSqrAbsComplexArray
from ScipyQuad import SphericalRestrictedIntegral
# from QuadPy import SphericalRestrictedIntegral
from VectorOperations import NullSpace5, HodgeDual, E3, randomLoop, GradEqFromMathematica, mdot
from parallel import parallel_map, ConstSharedArray, WritableSharedArray, print_debug, mp
from plot import Plot, PlotTimed, MakeDir, XYPlot, MakeNewDir
from Timer import MTimer as Timer

import warnings

from testMathematica import toMathematica

warnings.filterwarnings('ignore', 'The iteration is not making good progress')
#


MakeDir("plots")


def Sqr(v):  # Matrix (n,m) in sympy or in numpy
    return v.T.dot(v)


def test_list():
    test = [1, 2, 3] + [4, 5, 6, 7]
    test

'''
&&\vec \Phi_k =  \frac{1}{2} \csc \left(\frac{\beta }{2}\right) \left\{\cos (\alpha_k), \sin (\alpha_k) \vec w, i \cos \left(\frac{\beta }{2}\right)\right\};

 &&\alpha_{k+1} = \alpha_k + \sigma_k \beta;\\
    && \alpha_N = \alpha_0 =0;\\
    && \sigma_k^2 =1
    
    \beta = \frac {2 \pi p}{q}
    
    M = n q
'''
def Phi(alpha, beta):
    return np.array([cos(alpha), sin(alpha), cos(beta/2) * 1j]) / (2 * sin(beta/2))



def ImproveF1F2F3(F0, F1, F2, F3, F4):
    Dim = 3
    NF = 3

    def Eq(fi, fj):
        return [Sqr(fi - fj) - 1, (Sqr(fj) - Sqr(fi) - 1j) ** 2 + 1 - Sqr(fi + fj)]

    def pack(X):  # NF*Dim -> (NF,Dim)
        return np.array(X[0]).reshape((NF, Dim))

    def unpack(f1, f2, f3):  # (NF,Dim) ->NF*Dim
        return np.vstack([f1, f2, f3]).reshape(NF * Dim)

    def Eqs(*args):
        P = pack(args)
        f1 = P[0]
        f2 = P[1]
        f3 = P[2]
        eqs = np.array([Eq(F0, f1), Eq(f1, f2), Eq(f2, f3), Eq(f3, F4)]).reshape(8)
        extra1 = f2.T.dot(F0) - F2.T.dot(F0)
        # extra2 = f2.T.dot(F3) - F2.T.dot(F3)
        eqs = np.append(eqs, extra1)
        return eqs

    def Eqs2(*args):
        P = pack(args)
        f1 = P[0]
        f2 = P[1]
        f3 = P[2]
        eqs = np.array([Eq(F0, f1), Eq(f1, f2), Eq(f2, f3), Eq(f3, F4)]).reshape(8)
        extra1 = f2.T.dot(F4) - F2.T.dot(F4)
        # extra2 = f2.T.dot(F3) - F2.T.dot(F3)
        eqs = np.append(eqs, extra1)
        return eqs

    def Reqs(R, EE):
        C = np.zeros(int(len(R) / 2), dtype=complex)
        RealToComplexVec(R, C)
        eqs = EE(C)
        R = np.zeros(len(eqs) * 2, dtype=float)
        ComplexToRealVec(eqs, R)
        return R

    def f(*args):
        eqs = Eqs(*args)
        return SumSqrAbsComplexArray(eqs)

    def g(*args):
        return [z for z in Eqs(args)]

    def grad(f1, f2, f3):
        F = np.array([F0, f1, f2, f3, F4])
        X = GradEqFromMathematica(F)
        E9 = np.eye(9).astype(complex)
        dF2 = E9[3:6]
        x = mdot([F0, dF2])
        G = np.vstack([X, x])
        assert (G.shape == (9, 9))
        return G

    def Rgrad(R):
        C = np.zeros(int(len(R) / 2), dtype=complex)
        RealToComplexVec(R, C)
        P = pack([C])
        f1 = P[0]
        f2 = P[1]
        f3 = P[2]
        eqs = grad(f1, f2, f3)
        M = len(eqs) * 2
        R = np.zeros((M, M), dtype=float)
        ComplextoRealMat(eqs, R)
        return R

    ee = Eqs(unpack(F1, F2, F3))
    err0 = SumSqrAbsComplexArray(ee)

    X0 = unpack(F1, F2, F3)
    R0 = np.zeros(len(X0) * 2, dtype=float)
    ComplexToRealVec(X0, R0)
    test = grad(F1, F2, F3)
    rtest = Rgrad(R0)

    def reqs(R0):
        return Reqs(R0, Eqs)

    def reqs2(R0):
        return Reqs(R0, Eqs2)

    r = scipy.optimize.fsolve(reqs, R0, xtol=1e-9, fprime=Rgrad)
    x = np.zeros(len(X0), dtype=complex)
    RealToComplexVec(r, x)
    err1 = SumSqrAbsComplexArray(Eqs(x))
    if (err1 > 1e-23):
        print_debug("sum sqr error in equations reduced from ", err0, " to ", err1)
    P = pack([x])
    return P


class GroupFourierIntegral:
    def __init__(self):
        # mp.set_start_method('fork')
        s0 = np.matrix([[1, 0], [0, 1]])
        s1 = np.matrix([[0, 1], [1, 0]])
        s2 = np.matrix([[0, -1j], [1j, 0]])
        s3 = np.matrix([[1, 0], [0, -1]])
        self.Sigma = [s1, s2, s3]  # Pauli matrices
        self.Tau = [s0, 1j * s1, 1j * s2, 1j * s3]  # quaternions
        # self.session = WolframLanguageSession()
        # self.session.evaluate('Get["Notebooks/SphericalThetaIntegral.m"]')

    def TT(self, i, j, al, be):
        # O(3)_{i j} = q_a q_b tr_2 (s_i.Tau_a.s_j.Tau^H_b) = q_a q_b TT(i,j,a,b)
        return np.trace(mdot([self.Sigma[i], self.Tau[al], self.Sigma[j], self.Tau[be].H]))

    def GetRMatrix(self, X):
        # using quaternionic representation for tr_3 (O(3).X)
        # O(3)_{i j} = q_a q_b tr_2 (s_i.Tau_a.s_j.Tau^H_b)
        # Q = q_a Tau_a = q.Tau
        # O(3)_{i j} = tr_2 (s_i.Q.s_j.Q.H)
        #  V1.O(3).V2 =  tr_3( O(3).kron(V2,V1)) = O(3)_{i j} V2_j V1_i = tr_2 (HV1.Q.Hv2.Q.H)
        R = np.array(
            [
                [
                    np.sum([self.TT(i, j, al, be) * X[j, i] for i in range(3) for j in range(3)])
                    for al in range(4)]
                for be in range(4)]
        )
        return (R + R.T) * 0.5

    def ThetaIntegralsScipy(self, X1, X2, t0, t1, time_steps):
        rr = [self.GetRMatrix(X1), self.GetRMatrix(X2)]

        def TwoIntegrals(t):
            (r1, i1) = SphericalRestrictedIntegral(rr[0] / np.sqrt(t))
            (r2, i2) = SphericalRestrictedIntegral(rr[1] / np.sqrt(t))
            return (r1[0] + r2[0] + 1j * (i1[0] + i2[0])) / 2

        ans = parallel_map(TwoIntegrals, linspace(t0, t1, time_steps), 0)#mp.cpu_count())
        return ans

    def ThetaIntegralsMathematica(self, X1, X2, t0, t1, time_steps):
        R_string = toMathematica([self.GetRMatrix(X1), self.GetRMatrix(X2), [t0, t1, time_steps]])
        ans = self.session.evaluate('WW[' + R_string + ']')
        return [float(a.args[0]) + 1j * float(a.args[1]) for a in ans]

    def __del__(self):
        pass
        # self.session.terminate()


def test_GroupFourierIntegral():
    gfi = GroupFourierIntegral()
    r = np.random.normal(scale=0.1, size=30) + 1j * np.random.normal(scale=0.1, size=30)
    r = r.reshape((3, 10))
    X1 = r.dot(r.T)
    r = np.random.normal(scale=0.1, size=30) + 1j * np.random.normal(scale=0.1, size=30)
    r = r.reshape((3, 10))
    X2 = r.dot(r.T)
    with Timer("multi integrals in QuadPy"):
        test1 = gfi.ThetaIntegralsScipy(X1, X2, 1, 2, 100)
    print(test1)
    pass


class IterMoves():
    def __init__(self,N0, node_num):
        np.random.seed(node_num)
        self.node_num = node_num
        p = np.random.randint(1, N0)
        q = p + np.random.randint(1, N0)
        f = gcd(p, q)
        p = int(p/f)
        q = int(q/f)
        beta = 2 * pi * p / q
        n = np.random.randint(2,6)
        m = n + np.random.randint(2, 5)
        Nn =q * n
        Np = q*m
        self.M = Np + Nn
        sigmas = np.array([1]*Np  + [-1]* Nn,dtype = int)
        sigmas = np.random.permutation(sigmas)
        alphas = np.cumsum(sigmas) * beta
        self.Fstart = np.array([Phi(alphas[k], beta) for k in range(self.M)], complex)
        self.Frun = WritableSharedArray(self.Fstart)
        mp.set_start_method('fork')

    def GetSaveDirname(self):
        # os.path.join("plots", "curve_cycle_node." + str(cycle) + "." + str(node_num) + ".np")
        return os.path.join("plots", str(self.M))

    def GetSaveFilename(self, cycle):
        # os.path.join("plots", "curve_cycle_node." + str(cycle) + "." + str(node_num) + ".np")
        return os.path.join(self.GetSaveDirname(), "curve_cycle_node." + str(cycle) + "." + str(self.node_num) + ".np")

    def GetCLoopFilename(self):
        # os.path.join("plots", "curve_cycle_node." + str(cycle) + "." + str(node_num) + ".np")
        return os.path.join(self.GetSaveDirname(), "Cloop.np")

    def SaveCurve(self, cycle):
        self.Frun.tofile(self.GetSaveFilename(cycle))

    def MoveThreeVertices(self, zero_index, T, num_steps):
        M = self.M

        F0 = self.Frun[zero_index % M]
        F1 = self.Frun[(zero_index + 1) % M]
        F2 = self.Frun[(zero_index + 2) % M]
        F3 = self.Frun[(zero_index + 3) % M]
        F4 = self.Frun[(zero_index + 4) % M]
        Dim = 3
        NF = 3
        NC = Dim * NF
        NR = 2 * NC
        ZR = np.zeros(NR, dtype=float)
        ZC = np.zeros(NC, dtype=complex)
        X0 = ZR.copy()
        Y = np.vstack([F1, F2, F3]).reshape(NC)
        ComplexToRealVec(Y, X0)

        def g0(F1F2F3R):  # (NR,)
            F1F2F3C = ZC.copy()
            RealToComplexVec(F1F2F3R, F1F2F3C)  # (NC)
            Q = F1F2F3C.reshape((NF, Dim))
            f1 = Q[0]
            f2 = Q[1]
            f3 = Q[2]
            df1df2df3c = NullSpace5(np.array([F0, f1, f2, f3, F4]))  # NC,K
            k = df1df2df3c.shape[1]
            df1df2df3R = np.zeros((NR, 2 * k), dtype=float)
            ComplextoRealMat(df1df2df3c, df1df2df3R)
            return df1df2df3R

        K2 = g0(X0).shape[1]  # the dimension of real null space, to use in Wiener process

        def g(X, t):
            dXR = g0(X)
            return ReformatRealMatrix(dXR, K2)  # clip or truncate the derivative matrix

        tspan = np.linspace(0.0, T, num_steps)

        def f(x, t):
            return ZR

        #######TEST BEGIN
        noise = np.random.normal(size=K2)
        df1df2df3R = g(X0, 0).dot(noise)  # delta q = test.dot(q)
        df1df2df3c = ZC.copy()
        RealToComplexVec(df1df2df3R, df1df2df3c)
        #######TEST E#ND
        result = sdeint.itoEuler(f, g, X0, tspan)
        F1F2F3R = result[-1]
        F1F2F3C = ZC.copy()
        RealToComplexVec(F1F2F3R, F1F2F3C)  # (6)
        FF = F1F2F3C.reshape((NF, Dim))
        try:
            FF = ImproveF1F2F3(F0, FF[0], FF[1], FF[2], F4)
            # FF = ImproveF1F2F3(F0, F1, F2, F3, F4)
        except Exception as ex:
            print("failed to improve F1F2F3 ", ex)

        self.Frun[(zero_index + 1) % M, :] = FF[0]
        self.Frun[(zero_index + 2) % M, :] = FF[1]
        self.Frun[(zero_index + 3) % M, :] = FF[2]
        pass

    def CollectStatistics(self, t0, t1, time_steps):
        M = self.M
        try:
            CDir = np.fromfile(self.GetCLoopFilename(), float).reshape((-1, 3))
        except:
            C0 = randomLoop(M, 5)
            CDir = 0.5 * (C0 + np.roll(C0, -1, axis=0))
            CDir.tofile(self.GetCLoopFilename())
        CRev = CDir[::-1]
        pathnames = []
        GFI = GroupFourierIntegral()
        for filename in os.listdir(self.GetSaveDirname()):
            if filename.endswith(".np") and not ('psidata' in filename or 'Cloop' in filename):
                try:
                    splits = filename.split(".")
                    cycle = int(splits[-3])
                    node_num = int(splits[-2])
                    pathnames.append(os.path.join(self.GetSaveDirname(), filename))
                except Exception as ex:
                    print_debug(ex)
                pass
            pass
        pass

        def Psi(pathname):
            try:
                F = np.fromfile(pathname, dtype=complex)
                F = F.reshape((-1, 3))
                assert F.shape == (M, 3)  # (M by 3 complex tensor)
                Q = np.roll(F, -1, axis=0) - F  # (M by 3 complex tensor)
                X1 = np.dot(CDir.T, Q)  # (3 by 3 complex tensor for direct curve)
                X2 = np.dot(CRev.T, np.conjugate(Q))  # (3 by 3 complex tensor for reflected curve)
                return GFI.ThetaIntegralsScipy(X1, X2, t0, t1, time_steps)
            except Exception as ex:
                print(ex)
                return None

        result = parallel_map(Psi, pathnames, mp.cpu_count());
        psidata = np.array([x for x in result if x is not None], complex)
        psidata.tofile(
            os.path.join(self.GetSaveDirname(), "psidata." + str(t0) + "." + str(t1) + "." + str(time_steps) + ".np"))

    def PlotWilsonLoop(self, t0, t1, time_steps):
        # time_steps = int(time_steps/100)
        psidata = np.fromfile(
            os.path.join(self.GetSaveDirname(), "psidata." + str(t0) + "." + str(t1) + "." + str(time_steps) + ".np"),
            dtype=complex)
        psidata = psidata.reshape((-1, time_steps, 2))
        psidata = psidata.transpose((2, 1, 0))
        times = np.mean(psidata[0], axis=1)  # (time_steps,N)->time_steps
        psiR = np.mean(psidata[1].real, axis=1)  # (time_steps,N)->time_steps
        psiI = np.mean(psidata[1].imag, axis=1)  # (time_steps,N)->time_steps

        XYPlot([psiR, psiI], plotpath=os.path.join(self.GetSaveDirname(), "WilsonLoop.png"),
               scatter=False,
               title='Wilson Loop')

    def GetPathStats(self):
        pathnames = []
        for filename in os.listdir(self.GetSaveDirname()):
            if filename.endswith(".np") and not (('psidata' in filename) or ('Cloop' in filename)):
                try:
                    splits = filename.split(".")
                    cycle = int(splits[-3])
                    node_num = int(splits[-2])
                    pathnames.append(os.path.join(self.GetSaveDirname(), filename))
                except Exception as ex:
                    print_debug(ex)
                pass
            pass
        pass

        def EnstrophyCDF(pathname):
            M = self.M
            try:
                F = np.fromfile(pathname, dtype=complex)
                F = F.reshape((-1, 3))
                assert F.shape == (M, 3)  # (M by 3 complex tensor)
                Q = np.roll(F, -1, axis=0) - F  # (M by 3 complex tensor)
                FDF = np.sum(F ** 2, axis=1)
                FDQ = np.sum(F * Q, axis=1)
                return (FDQ ** 2 - FDF).real
            except Exception as ex:
                print(ex)
                return None

        result = parallel_map(EnstrophyCDF, pathnames, 0)#mp.cpu_count())
        cdfdata = np.array([x for x in result if x is not None], float).reshape(-1)
        cdfdata.tofile(
            os.path.join(self.GetSaveDirname(), "cdfdata.np"))

        def FractalDim(pathname):
            M = self.M
            try:
                F = np.fromfile(pathname, dtype=complex)
                F = F.reshape((-1, 3))
                assert F.shape == (M, 3)  # (M by 3 complex tensor)
                Q = np.roll(F, -1, axis=0) - F  # (M by 3 complex tensor)
                r = np.sqrt(np.sum(Q* np.conjugate(Q),axis=1).real)
                return r
            except Exception as ex:
                print(ex)
                return None

        result = parallel_map(FractalDim, pathnames,0)# mp.cpu_count())
        fracdata = np.array([x for x in result if x is not None], float)
        fracdata.tofile(
            os.path.join(self.GetSaveDirname(), "fracdata.np"))
    def PlotEnstropyDistribution(self):
        cdfdata = np.fromfile(
            os.path.join(self.GetSaveDirname(), "cdfdata.np"), dtype=float)
        from plot import RankHistPos
        RankHistPos(cdfdata, plotpath=os.path.join(self.GetSaveDirname(), "cdfdata.png"), name="Enstrophy CDF", logx=True, logy=True)

    def PlotFractalDimension(self):
        fracdata = np.fromfile(
            os.path.join(self.GetSaveDirname(), "fracdata.np"), dtype=float).reshape(-1)
        from plot import RankHistPos
        RankHistPos(fracdata,logx=True, logy=True, plotpath=os.path.join(self.GetSaveDirname(), "fracdata.png"), name="StepSize CFD",var_name='step')


def runIterMoves(num_vertices=100,num_cycles=10, T=1.0, num_steps=1000,
                 t0=1., t1=10., time_steps=100,
                 node=0, NewRandomWalk=False, plot=True):
    mover = IterMoves(num_vertices, node_num=node)
    def MoveThree(zero_index):
        try:
            mover.MoveThreeVertices(zero_index, T, num_steps)
        except Exception as ex:
            print_debug("Exception ", ex)
    M = mover.M
    MoveThree(0)

    if NewRandomWalk:
        MakeNewDir(mover.GetSaveDirname())
        mess = "ItoProcess with N=" + str(M) + " and " + str(num_cycles) + " cycles each with " + str(
            num_steps) + " Ito steps"
        print_debug("starting " + mess)
        with Timer(mess):
            for cycle in range(num_cycles):
                for zero_index in range(4):
                    parallel_map(MoveThree, range(zero_index, M + zero_index, 4), mp.cpu_count())
                    print_debug("after cycle " + str(cycle) + " zero index " + str(zero_index))
                pass
                mover.SaveCurve(cycle)
                print("after saving curve at cycle " + str(cycle))
            pass
            print("all cycles done " + str(cycle))
        pass
    pass
    if plot:
        mover.GetPathStats()
        mover.PlotEnstropyDistribution()
        mover.PlotFractalDimension()
    else:
        mover.CollectStatistics(t0, t1, time_steps)
        mover.PlotWilsonLoop(t0, t1, time_steps)


def test_IterMoves():
    runIterMoves(num_vertices=100, num_cycles=100, T=1., num_steps=10000,
                 t0=1, t1=10, time_steps=100,
                 node=0, NewRandomWalk=True, plot=False)


def testarrays():
    F = np.arange(10) -0.5* 1j * np.arange(10)
    test = np.roll(F, -1, axis=0) - F
    y =np.abs(test)



if __name__ == '__main__':
    import argparse
    import logging

    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=1000)
    parser.add_argument('-C', type=int, default=100)
    parser.add_argument('-S', type=int, default=10000)
    parser.add_argument('-TS', type=int, default=1000)
    parser.add_argument('-P', type=int, default=0)
    parser.add_argument('-debug', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-plot', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-new-random-walk', action=argparse.BooleanOptionalAction, default=False)
    A = parser.parse_args()
    if A.debug: logging.basicConfig(level=logging.DEBUG)
    # print(A)
    runIterMoves(num_vertices=A.N, num_cycles=A.C, num_steps=A.S, time_steps=A.TS, node=A.P,
                 NewRandomWalk=A.new_random_walk, plot=A.plot)
