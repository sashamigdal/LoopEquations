import math
from os.path import split

import scipy


import numpy as np
import os, sys
from numpy import cos, sin, pi
from scipy.linalg import pinvh

from scipy.stats import ortho_group
import mpmath
from mpmath import hyp0f1, hyp1f2
import sdeint
import multiprocessing as mp

from ComplexToReal import ComplexToRealVec, RealToComplexVec, ComplextoRealMat, ReformatRealMatrix, MaxAbsComplexArray, \
    SumSqrAbsComplexArray
from VectorOperations import NullSpace5, HodgeDual, E3, randomLoop
from parallel import parallel_map, ConstSharedArray, WritableSharedArray, print_debug
from plot import Plot, PlotTimed, MakeDir, XYPlot, MakeNewDir
from Timer import MTimer as Timer

from mpmath import mp as mpm

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
#


MakeDir("plots")


def Sqr(v):  # Matrix (n,m) in sympy or in numpy
    return v.T.dot(v)

def test_list():
    test = [1, 2, 3] + [4, 5, 6, 7]
    test


def ff(k, M):
    delta = pi / M
    return np.array([cos(2 * k * delta), sin(2 * k * delta), cos(delta) * 1j]) / (2 * sin(delta))


def gamma_formula(u):
    assert u.ndim == 1
    d = u.shape[0]
    V = np.vstack([u.real, u.imag]).reshape((2, d))
    Q = pinvh(V.T.dot(V))
    V1 = np.dot(Q, V.T)  # (d,2)
    return V1[:, 1] + V1[:, 0] * 1j


def VecGamma(U, Ans):
    Ans[:] = np.vstack([gamma_formula(u) for u in U])


def testNullSpace():
    # NS = NullSpace3(ff(0, 10), ff(1, 10), ff(2, 10))
    #NS = NullSpace4(ff(0, 10), ff(1, 10), ff(2, 10), ff(3, 10))
    NS = NullSpace5(np.array([ff(0, 10), ff(1, 10), ff(2, 10),ff(3, 10),ff(4, 10)],dtype=complex))
    pass

def ImproveF1F2F3(F0, F1, F2, F3, F4):
    Dim = 3
    NF = 3

    def Eq(fi,fj):
        return [Sqr(fi-fj) -1,(Sqr(fj) - Sqr(fi) -1j)**2 + 1 - Sqr(fi+fj)]


    def pack(X): # NF*Dim -> (NF,Dim)
        return np.array(X[0]).reshape((NF,Dim))

    def unpack(f1, f2, f3):# (NF,Dim) ->NF*Dim
        return np.vstack([f1,f2,f3]).reshape(NF*Dim)

    def Eqs(*args):
        P = pack(args)
        f1 = P[0]
        f2 = P[1]
        f3 = P[2]
        eqs = np.array([Eq(F0, f1), Eq(f1, f2), Eq(f2, f3), Eq(f3, F4)]).reshape(8)
        extra1 = f2.T.dot(F0) - F2.T.dot(F0)
        # extra2 = f2.T.dot(F3) - F2.T.dot(F3)
        eqs = np.append(eqs,extra1)
        return eqs

    def Reqs(R):
        C = np.zeros(int(len(R)/2), dtype=complex)
        RealToComplexVec(R,C)
        eqs = Eqs(C)
        R = np.zeros(len(eqs)*2,dtype=float)
        ComplexToRealVec(eqs,R)
        return R

    def f(*args):
        eqs = Eqs(*args)
        return SumSqrAbsComplexArray(eqs)

    def g(*args):
        return [z for z in Eqs(args)]

    pass
    ee = Eqs(unpack(F1,F2,F3))
    err0 = SumSqrAbsComplexArray(ee)
    mpm.dps = 40
    X0 = unpack(F1, F2,F3)
    R0 = np.zeros(len(X0) * 2, dtype=float)
    ComplexToRealVec(X0, R0)
    test = Reqs(R0)
    r = scipy.optimize.fsolve(Reqs,R0,xtol=1e-9)
    x = np.zeros(len(X0),dtype=complex)
    RealToComplexVec(r,x)
    err1 = SumSqrAbsComplexArray(Eqs(x))
    if(err1 > 1e-20):
       print("sum sqr error in equations reduced from ", err0, " to ", err1 )
       r = scipy.optimize.fsolve(Reqs, r, xtol=1e-14)
       x = np.zeros(len(X0), dtype=complex)
       RealToComplexVec(r, x)
       err2 = SumSqrAbsComplexArray(Eqs(x))
       print("sum sqr error in equations further reduced from ", err1, " to ", err2)
    P = pack([x])
    return  P

class IterMoves():
    def __init__(self, M):
        self.M = M
        self.Fstart = np.array([ff(k, M) for k in range(M)], complex)
        self.Frun = WritableSharedArray(self.Fstart)

    def GetSaveDirname(self):
        # os.path.join("plots", "curve_cycle_node." + str(cycle) + "." + str(node_num) + ".np")
        return os.path.join("plots", str(self.M))

    def GetSaveFilename(self, cycle, node_num):
        # os.path.join("plots", "curve_cycle_node." + str(cycle) + "." + str(node_num) + ".np")
        return os.path.join(self.GetSaveDirname(), "curve_cycle_node." + str(cycle) + "." + str(node_num) + ".np")

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
            print("failed to improve F1F2F3 ",ex)

        self.Frun[(zero_index + 1) % M,:] = FF[0]
        self.Frun[(zero_index + 2) % M,:] = FF[1]
        self.Frun[(zero_index + 3) % M,:] = FF[2]
        pass

    def SaveCurve(self, cycle, node_num):
        self.Frun.tofile(self.GetSaveFilename(cycle, node_num))

    def CollectStatistics(self, C0, t0, t1, time_steps):
        CDir = 0.5 * (C0 + np.roll(C0, 1, axis=0))
        CRev = CDir[::-1]
        M = self.M
        mpm.dps =12
        pathnames = []
        for filename in os.listdir(self.GetSaveDirname()):
            if filename.endswith(".np") and not ('psidata' in filename):
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
        # time_steps = int(time_steps/100)
        def Psi(pathname):
            ans = []
            try:
                F = np.fromfile(pathname, dtype=complex)
                F = F.reshape((-1, 3))
                assert F.shape == (M, 3)  # (M by 3 complex tensor)
                Q = np.roll(F, 1, axis=0) - F  # (M by 3 complex tensor)
                X = np.dot(CDir.T, Q)  # (3 by 3 complex tensor for direct curve)
                z0 = np.sqrt(np.trace(X.dot(X.T)))
                X = np.dot(CRev.T, np.conjugate(Q))  # (3 by 3 complex tensor for reflected curve)
                z1 = np.sqrt(np.trace(X.dot(X.T)))
                for t in np.linspace(t0, t1, time_steps):
                    psi = 0. + 0.j
                    for z in (z0 / np.sqrt(t), z1 / np.sqrt(t)):
                        if z.imag != 0:
                            z *= np.sign(z.imag)
                            # \frac{1}{2} \, _0F_1\left(;2;-\frac{z^2}{4}\right)+\frac{2 i z \, _1F_2\left(1;\frac{3}{2},\frac{5}{2};-\frac{z^2}{4}\right)}{3 \pi }
                            psi += 1. / 4. * hyp0f1(2, -z * z / 4) + 2j * z / (3 * pi) * hyp1f2(1, 3. / 2, 5. / 2, -z * z / 4)
                        else:
                            psi += 1. / 4. * hyp0f1(2, -z * z / 4)
                        pass
                    pass
                    ans.append([t, psi])
                return ans
            except Exception as ex:
                print_debug(ex)
                return None

        result = parallel_map(Psi, pathnames, mp.cpu_count())
        psidata = np.array([x for x in result if x is not None], complex)
        psidata.tofile(
            os.path.join(self.GetSaveDirname(), "psidata." + str(t0) + "." + str(t1) + "." + str(time_steps) + ".np"))

    def PlotWilsonLoop(self, t0, t1, time_steps):
        # time_steps = int(time_steps/100)
        psidata = np.fromfile(
            os.path.join(self.GetSaveDirname(), "psidata." + str(t0) + "." + str(t1) + "." + str(time_steps) + ".np"),
            dtype=complex)
        psidata = psidata.reshape((-1, time_steps, 2))
        psidata = psidata.transpose((2,1, 0))
        times = np.mean(psidata[0], axis=1)  # (time_steps,N)->time_steps
        psiR = np.mean(psidata[1].real, axis=1)  # (time_steps,N)->time_steps
        psiI = np.mean(psidata[1].imag, axis=1)  # (time_steps,N)->time_steps

        XYPlot([psiR, psiI], plotpath=os.path.join(self.GetSaveDirname(), "WilsonLoop.png"),
               scatter=True,
               title='Wilson Loop')


def runIterMoves(num_vertices=100, num_cycles=10, T=1.0, num_steps=1000,
                 t0=1., t1=10., time_steps=100,
                 node=0, NewRandomWalk=False, plot = True):
    M = num_vertices
    mover = IterMoves(M)
    mp.set_start_method('fork')
    def MoveThree(zero_index):
        try:
            mover.MoveThreeVertices(zero_index, T, num_steps)
        except Exception as ex:
          print_debug("Exception ", ex)

    MoveThree(0)
    C0 = randomLoop(M, 5)
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
                mover.SaveCurve(cycle, node)
                print("after saving curve at cycle " + str(cycle))
            pass
            print("all cycles done " + str(cycle))
        pass
    pass
    mover.CollectStatistics(C0, t0, t1, time_steps)
    if plot:
        mover.PlotWilsonLoop(t0, t1, time_steps)


def test_IterMoves():
    runIterMoves(num_vertices=500, num_cycles=100, T=1., num_steps=10000,
                 t0=1, t1=10, time_steps=100,
                 node=0, NewRandomWalk=True, plot=False)


def testFF():
    print_debug(ff(3, 10))


def test_gamma_formula():
    u = np.array(
        [[1 + 1j, 2 + 2j, 3 + 3j], [1 - 1j, 2 - 2j, 3 - 3j], [5 - 6j, 7 - 8j, 9 - 10j], [-5 - 6j, -7 - 8j, -9 - 10j]])
    gamma = np.zeros_like(u)
    gamma = np.vstack([gamma_formula(x) for x in u])
    gamma


def testtensorDot():
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    b = 10 * a
    c = np.tensordot(a, b, axes=([2], [2]))
    a1 = a.reshape((4, 3))
    b1 = b.reshape((4, 3))
    c1 = a1.dot(b1.T).reshape((2, 2, 2, 2))
    pass


def testPairs():
    K = 2
    M = 3
    S = [2]
    pairs = np.array(np.meshgrid(np.arange(K), np.arange(M))).T.reshape((-1, 2))
    T = np.concatenate(([K, M], S))
    test = pairs.reshape(T)
    test


def testDual():
    E = HodgeDual(np.array([1 + 1j, 1 + 1j, 1 + 1j]))
    E


if __name__ == '__main__':
    import argparse
    import logging
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int,default=1000)
    parser.add_argument('-C', type=int,default=100)
    parser.add_argument('-S', type=int,default=10000)
    parser.add_argument('-TS', type=int,default=1000)
    parser.add_argument('-P', type=int,default=0)
    parser.add_argument('-debug', action = argparse.BooleanOptionalAction,default=False)
    parser.add_argument('-plot', action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('-new-random-walk', action=argparse.BooleanOptionalAction, default=False)
    A = parser.parse_args()
    if A.debug: logging.basicConfig(level=logging.DEBUG)
    # print(A)
    runIterMoves(num_vertices=A.N, num_cycles=A.C, num_steps=A.S,time_steps=A.TS, node=A.P,
                 NewRandomWalk=A.new_random_walk, plot=A.plot)
