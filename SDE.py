import math
from os.path import split

import numpy as np
import os, sys
from numpy import cos, sin, pi
from scipy.linalg import pinvh

from scipy.stats import ortho_group
from mpmath import hyp0f1, hyp1f2
import sdeint
import multiprocessing as mp

from ComplexToReal import ComplexToRealVec, RealToComplexVec, ComplextoRealMat, ReformatRealMatrix, MaxAbsComplexArray
from VectorOperations import NullSpace4, HodgeDual, E3
from parallel import parallel_map, ConstSharedArray, WritableSharedArray
from plot import Plot, PlotTimed, MakeDir, XYPlot, MakeNewDir
from Timer import MTimer as Timer
import logging

logger = logging.Logger("debug")

MakeDir("plots")


def ff(k, M):
    delta = pi / M
    return np.array([cos(2 * k * delta), sin(2 * k * delta), cos(delta) * 1j]) / (2 * sin(delta))


def cc(k, M):
    delta = pi / M
    return np.array([cos(2 * k * delta), sin(2 * k * delta), 0], dtype=float)


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

        ZR = np.zeros(12, dtype=float)
        ZC = np.zeros(6, dtype=complex)
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
        result = sdeint.itoSRI2(f, g, X0, tspan)
        F1F2R = result[-1]
        F1F2C = ZC.copy()
        RealToComplexVec(F1F2R, F1F2C)  # (6)
        FF = F1F2C.reshape(2, 3)
        q0 = FF[0] - F0
        q1 = FF[1] - FF[0]
        q2 = F3 - FF[1]
        test = np.array([q0.dot(q0) - 1, q2.dot(q2) - 1])
        err = MaxAbsComplexArray(test)
        if err > 1e-5:
            print("correcting large error in q_i^2", err)
            q0 /= np.sqrt(q0.dot(q0))
            q2 /= np.sqrt(q2.dot(q2))
            FF[0] = F0 + q0
            FF[1] = F3 - q2

        if False:  # just for test
            np.set_printoptions(linewidth=np.inf)
            print(f"F0,F3 :\n{F0}\n{F3}")
            print(f"F1(0),F1(t) :\n{F1}\n{FF[0]}")
            print(f"F2(0),F2(t) :\n{F2}\n{FF[1]}")
        pass
        self.Frun[(zero_index + 1) % M][:] = FF[0]
        self.Frun[(zero_index + 2) % M][:] = FF[1]
        pass

    def SaveCurve(self, cycle, node_num):
        self.Frun.tofile(self.GetSaveFilename(cycle, node_num))

    def CollectStatistics(self):
        pass

    def PlotResults(self, C0):
        C1 = C0 + np.roll(C0, 1, axis=0)
        C1 = 0.5 * C1
        M = self.M

        def Psi(pathname):
            F = np.fromfile(pathname, dtype=complex).reshape(-1, 3)
            assert F.shape == (M, 3)  # (M by 3 complex tensor)
            q = np.roll(F, 1, axis=0) - F  # (M by 3 complex tensor)
            X = np.dot(C1.T, q)  # (3 by 3 complex tensor)
            z = np.sqrt(np.trace(X.dot(X.T)))
            if z.imag != 0:
                z *= np.sign(z.imag)
                # \frac{1}{2} \, _0F_1\left(;2;-\frac{z^2}{4}\right)+\frac{2 i z \, _1F_2\left(1;\frac{3}{2},\frac{5}{2};-\frac{z^2}{4}\right)}{3 \pi }
                ans = 1 / 2 * hyp0f1(2, -z * z / 4) + 2j * z / (3 * pi) * hyp1f2(1, 3. / 2, 5. / 2, -z * z / 4)
            else:
                ans = 1. / 2 * hyp0f1(2, -z * z / 4)
            return complex(ans)

        pathnames = []
        for filename in os.listdir(self.GetSaveDirname()):
            if filename.endswith(".np"):
                try:
                    splits = filename.split(".")
                    cycle = int(splits[-3])
                    node_num = int(splits[-2])
                    pathnames.append([os.path.join(self.GetSaveDirname(), filename)])
                except Exception as ex:
                    print(ex)

        emap = parallel_map(Psi, pathnames, mp.cpu_count())

        psidata = np.array(list(emap), complex)
        XYPlot([psidata.real, psidata.imag], plotpath=os.path.join(self.GetSaveDirname(), "WilsonLoop.png"),
               scatter=True,
               title='Wilson Loop')


def runIterMoves(num_vertices=100, num_cycles=10, T=0.1, num_steps=1000, node=0, NewRandomWalk=False):
    M = num_vertices
    mover = IterMoves(M)

    def MoveTwo(zero_index):
        try:
            mover.MoveTwoVertices(zero_index, T, num_steps)
        except Exception as ex:
            print("Exception ", ex)

    MoveTwo(0)
    if NewRandomWalk:
        MakeNewDir(mover.GetSaveDirname())
        mess = "ItoProcess with N=" + str(M) + " and " + str(num_cycles) + " cycles each with " + str(
            num_steps) + " Ito steps"
        logger.debug("starting " + mess, exc_info=1)
        with Timer(mess):
            for cycle in range(num_cycles):
                for zero_index in range(3):
                    parallel_map(MoveTwo, range(zero_index, M + zero_index, 3), mp.cpu_count())
                    logger.debug("after cycle " + str(cycle) + " zero index " + str(zero_index))
                pass
                mover.SaveCurve(cycle, node)
                logger.debug("after saving curve at cycle " + str(cycle))
            pass
            logger.debug("all cycles done " + str(cycle))
        pass
        pass
    pass
    C0 = np.array([cc(k, M) for k in range(M)])
    m = ortho_group.rvs(dim=3)
    C0 = C0.dot(m)
    mover.PlotResults(C0)


def test_IterMoves():
    runIterMoves(num_vertices=300, num_cycles=10, T=0.1, num_steps=1000, node=0, NewRandomWalk=True)


def testFF():
    print(ff(3, 10))


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
        print("test iter moves")
        test_IterMoves()
