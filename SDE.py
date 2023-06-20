import math
from os.path import split

import scipy

DEBUG = True
import numpy as np
import os, sys
from numpy import cos, sin, pi
from scipy.linalg import pinvh

from scipy.stats import ortho_group
import mpmath
from mpmath import hyp0f1, hyp1f2
import sdeint
import multiprocessing as mp

from ComplexToReal import ComplexToRealVec, RealToComplexVec, ComplextoRealMat, ReformatRealMatrix, MaxAbsComplexArray
from VectorOperations import NullSpace4, HodgeDual, E3, randomLoop
from parallel import parallel_map, ConstSharedArray, WritableSharedArray, print_debug
from plot import Plot, PlotTimed, MakeDir, XYPlot, MakeNewDir
from Timer import MTimer as Timer
import logging
from sympy import MatrixSymbol, Matrix, I, series, trace, diff

from mpmath import mp as mpm
#
mpm.dps = 30
prec = 20
logger = logging.Logger("debug")

MakeDir("plots")


def Sqr(v):  # Matrix (n,m) in sympy or in numpy
    return v.T.dot(v)


class SolveTwoVertices:  # using sympy
    def __init__(self, F0, F1, F2, F3):
        self.F0 = Matrix(F0.reshape(3,1))
        self.F1 = Matrix(F1.reshape(3,1))
        self.F2 = Matrix(F2.reshape(3,1))
        self.F3 = Matrix(F3.reshape(3,1))

    pass

    # (F{k+1} - Fk)^2 =1
    # (F{k+1}^2 - Fk^2 -I)^2 = (F{k+1} + Fk)^2 -1
    @staticmethod
    def Dual(v):  # first index is a vector index
        x = np.array(v).astype(complex).reshape(3)
        y = E3.dot(x)
        return Matrix(y)

    @staticmethod
    def Eq( fi, fj):
        si = Sqr(fi)
        sj= Sqr(fj)
        sij = Sqr(fi + fj)
        return ( sj- si- I) ** 2 + 1 - sij

    @staticmethod
    def pack( X):
        y = np.array(X, float).reshape(2, 6)
        return np.array(y[0] + 1j * y[1], complex).reshape(2, 3)

    @staticmethod
    def unpack(f1, f2):
        x = np.vstack([f1, f2]).reshape(6)
        return np.vstack([x.real, x.imag]).reshape(12).astype(float)

    def Eqs(self, *args):
        P = self.pack(args)
        f1 = P[0]
        f2 = P[1]
        return [self.Eq(self.F0, f1), self.Eq(f1, f2), self.Eq(f2, self.F3)]

    def GradientMatrix(self):
        q0 = self.F1 - self.F0
        q1 = self.F2 - self.F1
        q2 = self.F3 - self.F2
        dt0 = MatrixSymbol('dt0', 3, 1)
        dt1 = MatrixSymbol('dt1', 3, 1)
        dt2 = MatrixSymbol('dt2', 3, 1)
        Z3 = Matrix(np.zeros((3, 3), dtype=complex))
        qd0 = self.Dual(q0)
        qd1 = self.Dual(q1)
        qd2 = self.Dual(q2)
        dq0 = qd0 * dt0  # (3 by NZ) infinitesimal rotation opf q0
        dq1 = qd1 * dt1  # (3 by NZ)infinitesimal rotation opf q1
        dq2 = qd2 * dt2 # (3 by NZ)infinitesimal rotation opf q2
        f1 = self.F1 + dq0
        f2 = self.F3 - dq2
        eqs = [self.Eq(self.F0, f1), self.Eq(f1, f2), self.Eq(f2, self.F3), dq0 + dq1 + dq2]
        lineqs = [series(eq, dt0, Z3, 1) for eq in eqs]
        pass
    def Correction(self, tol=1e-16):
        def f(*args):
            return sum([np.abs(eq) ** 2 for eq in self.Eqs(*args)])

        pass
        X0 = self.unpack(self.F1, self.F2)
        ee = self.Eqs(X0)
        err0 = MaxAbsComplexArray(np.array(ee, dtype=complex))
        res = scipy.optimize.minimize(f, X0, tol=tol)
        ee1 = self.Eqs(res.x)
        err1 = MaxAbsComplexArray(np.array(ee1, dtype=complex))
        P = self.pack(res.x)
        self.F1 = P[0]
        self.F2 = P[1]

def test_SolveTwoVertices():
    FF = randomLoop(10,2) + 1j * randomLoop(10,2)
    solver = SolveTwoVertices(FF[0],FF[1],FF[2],FF[3])
    solver.GradientMatrix()
def test_list():
    test = [1, 2, 3] + [4, 5, 6, 7]
    test


def ff(k, M):
    delta = pi / M
    return np.array([cos(2 * k * delta), sin(2 * k * delta), cos(delta) * 1j]) / (2 * sin(delta))


def gamma_formula(u):
    assert u.ndim == 1
    d = u.shape[0]
    V = np.vstack([u.real, u.imag]).reshape(2, d)
    Q = pinvh(V.T.dot(V))
    V1 = np.dot(Q, V.T)  # (d,2)
    return V1[:, 1] + V1[:, 0] * 1j


def VecGamma(U, Ans):
    Ans[:] = np.vstack([gamma_formula(u) for u in U])


def testNullSpace():
    # NS = NullSpace3(ff(0, 10), ff(1, 10), ff(2, 10))
    NS = NullSpace4(ff(0, 10), ff(1, 10), ff(2, 10), ff(3, 10))
    pass

def ImproveF1F2(F0, F1, F2, F3):
    #(F{k+1} - Fk)^2 =1
    # (F{k+1}^2 - Fk^2 -I)^2 = (F{k+1} + Fk)^2 -1

    def Eq(fi,fj):
        return [Sqr(fi-fj) - 1,
                (Sqr(fj) - Sqr(fi) -1j)**2 + 1 - Sqr(fi+fj)]


    def pack(X):
        y = np.array(X,float).reshape(2,6)
        return np.array(y[0]+ 1j * y[1],complex).reshape(2,3)

    def unpack(f1, f2):
        x= np.vstack([f1,f2]).reshape(6)
        retval = np.vstack([x.real,x.imag]).reshape(12).astype(float)
        return retval

    def Eqs(*args):
        P = pack(args)
        f1 = P[0]
        f2 = P[1]
        return np.array([Eq(F0, f1), Eq(f1, f2), Eq(f2, F3)]).reshape(-1)

    def f(*args):
        eqs = Eqs(*args)
        return np.conjugate(eqs).dot(eqs).real

    def g(*args):
        eqs = Eqs(args)
        return [mpmath.mpc(z) for z in eqs]

    pass
    ee = Eqs(unpack(F1,F2))
    err0 = MaxAbsComplexArray(ee)
    # res = scipy.optimize.minimize(f,unpack(F1,F2),tol=1e-16)
    # ee1 = Eqs(res.x)
    # err1 = MaxAbsComplexArray(np.array(ee1,dtype=complex))complex
    # if(err0 > 1e-6):
    #     print("error reduced from ", err0, " to ", err1 )
    # P = pack(res.x)

    initial_value = [mpmath.mpf(x) for x in unpack(F1, F2)]
    x = mpm.findroot(g, initial_value, solver='bisect')
    err1 = MaxAbsComplexArray(Eqs(x))
    if(err0 > 1e-6):
       print("error reduced from ", err0, " to ", err1 )
    P = pack(x)
    return  P[0],P[1]

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

    def MoveTwoVertices(self, zero_index, T, num_steps):
        M = self.M

        F0 = self.Frun[zero_index % M]
        F1 = self.Frun[(zero_index + 1) % M]
        F2 = self.Frun[(zero_index + 2) % M]
        F3 = self.Frun[(zero_index + 3) % M]
        NF = 2
        NC = 3 * NF
        NR = 2 * NC
        ZR = np.zeros(NR, dtype=float)
        ZC = np.zeros(NC, dtype=complex)
        X0 = ZR.copy()
        Y = np.vstack([F1, F2]).reshape(6)
        ComplexToRealVec(Y, X0)

        def g0(F1F2R):  # (12,)
            F1F2C = ZC.copy()
            RealToComplexVec(F1F2R, F1F2C)  # (6)
            Q = F1F2C.reshape(2, 3)
            f1 = Q[0]
            f2 = Q[1]
            dF1dF2c = NullSpace4(F0, f1, f2, F3)  # 6,K
            k = dF1dF2c.shape[1]
            dF1dF2R = np.zeros((12, 2 * k), dtype=float)
            ComplextoRealMat(dF1dF2c, dF1dF2R)
            return dF1dF2R

        K2 = g0(X0).shape[1]  # the dimension of real null space, to use in Wiener process

        def g(X, t):
            dXR = g0(X)
            return ReformatRealMatrix(dXR, K2)  # clip or truncate the derivative matrix

        tspan = np.linspace(0.0, T, num_steps)

        def f(x, t):
            return ZR

        #######TEST BEGIN
        noise = np.random.normal(size=K2)
        df1df2R = g(X0, 0).dot(noise)  # delta q = test.dot(q)
        dF1dF2C = ZC.copy()
        RealToComplexVec(df1df2R, dF1dF2C)
        #######TEST E#ND
        result = sdeint.itoEuler(f, g, X0, tspan)
        F1F2R = result[-1]
        F1F2C = ZC.copy()
        RealToComplexVec(F1F2R, F1F2C)  # (6)
        FF = F1F2C.reshape(2, 3)
        q0 = FF[0] - F0
        q1 = FF[1] - FF[0]
        q2 = F3 - FF[1]
        test = np.array([q0.dot(q0) - 1, q1.dot(q1) - 1, q2.dot(q2) - 1])
        err = MaxAbsComplexArray(test)
        if err >1e-4:  # just for test
            print("err =",err)
            # np.set_printoptions(linewidth=np.inf)
            # print_debug("(F2-F1)^2-1 = ", (F2 - F1).dot(F2 - F1) - 1)
            # print_debug(f"F0,F3 :\n{F0}\n{F3}")
            # print_debug(f"F1(0),F1(t) :\n{F1}\n{FF[0]}")
            # print_debug(f"F2(0),F2(t) :\n{F2}\n{FF[1]}")
        #restore the imnprove function here
        F1, F2 = ImproveF1F2(F0, FF[0], FF[1], F3)
        self.Frun[(zero_index + 1) % M][:] = F1
        self.Frun[(zero_index + 2) % M][:] = F2
        pass

    def SaveCurve(self, cycle, node_num):
        self.Frun.tofile(self.GetSaveFilename(cycle, node_num))

    def CollectStatistics(self, C0, t0, t1, time_steps):
        CDir = 0.5 * (C0 + np.roll(C0, 1, axis=0))
        CRev = CDir[::-1]
        M = self.M
        pathnames = []
        for filename in os.listdir(self.GetSaveDirname()):
            if filename.endswith(".np"):
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
            ans = []
            try:
                F = np.fromfile(pathname, dtype=complex).reshape(-1, 3)
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
                            psi += 1. / 4. * hyp0f1(2, -z * z / 4) + 2j * z / (3 * pi) * hyp1f2(1, 3. / 2, 5. / 2,
                                                                                                -z * z / 4)
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
        psidata = np.fromfile(
            os.path.join(self.GetSaveDirname(), "psidata." + str(t0) + "." + str(t1) + "." + str(time_steps) + ".np"),
            dtype=complex).reshape(-1, time_steps, 2)

        psidata = psidata.transpose((2, 1, 0))  # (2,time_steps, N)
        times = np.mean(psidata[0], axis=1)  # (time_steps,N)->time_steps
        psiR = np.mean(psidata[1].real, axis=1)  # (time_steps,N)->time_steps
        psiI = np.mean(psidata[1].imag, axis=1)  # (time_steps,N)->time_steps

        XYPlot([psiR, psiI], plotpath=os.path.join(self.GetSaveDirname(), "WilsonLoop.png"),
               scatter=True,
               title='Wilson Loop')


def runIterMoves(num_vertices=100, num_cycles=10, T=0.1, num_steps=1000,
                 t0=1, t1=10, time_steps=100,
                 node=0, NewRandomWalk=False):
    M = num_vertices
    mover = IterMoves(M)
    mp.set_start_method('fork')
    def MoveTwo(zero_index):
        try:
            mover.MoveTwoVertices(zero_index, T, num_steps)
        except Exception as ex:
            print_debug("Exception ", ex)

    MoveTwo(0)
    C0 = randomLoop(M, 5)
    if NewRandomWalk:
        MakeNewDir(mover.GetSaveDirname())
        mess = "ItoProcess with N=" + str(M) + " and " + str(num_cycles) + " cycles each with " + str(
            num_steps) + " Ito steps"
        print_debug("starting " + mess)
        with Timer(mess):
            for cycle in range(num_cycles):
                for zero_index in range(3):
                    parallel_map(MoveTwo, range(zero_index, M + zero_index, 3), mp.cpu_count())
                    print_debug("after cycle " + str(cycle) + " zero index " + str(zero_index))
                pass
                mover.SaveCurve(cycle, node)
                print_debug("after saving curve at cycle " + str(cycle))
            pass
            print_debug("all cycles done " + str(cycle))
        pass
    pass
    mover.CollectStatistics(C0, t0, t1, time_steps)
    mover.PlotWilsonLoop(t0, t1, time_steps)


def test_IterMoves():
    runIterMoves(num_vertices=300, num_cycles=100, T=1, num_steps=100,
                 t0=1, t1=1, time_steps=100,
                 node=0, NewRandomWalk=True)


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
    a1 = a.reshape(4, 3)
    b1 = b.reshape(4, 3)
    c1 = a1.dot(b1.T).reshape(2, 2, 2, 2)
    pass


def testPairs():
    K = 2
    M = 3
    S = [2]
    pairs = np.array(np.meshgrid(np.arange(K), np.arange(M))).T.reshape(-1, 2)
    T = np.concatenate(([K, M], S))
    test = pairs.reshape(*T)
    test


def testDual():
    E = HodgeDual(np.array([1 + 1j, 1 + 1j, 1 + 1j]))
    E


if __name__ == '__main__':
    if len(sys.argv) == 3:
        N = int(sys.argv[1])
        P = int(sys.argv[2])
        runIterMoves(num_vertices=N, num_cycles=100, num_steps=1000, T=1, node=P, NewRandomWalk=True)
    else:
        print_debug("test iter moves")
        test_IterMoves()
