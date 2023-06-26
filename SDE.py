import math
from os.path import split

import scipy

import numpy as np
import os, sys
from numpy import cos, sin, pi
from scipy.linalg import pinvh

from scipy.stats import ortho_group
import mpmath as mpm
from mpmath import hyp0f1, hyp1f2
import sdeint
import multiprocessing as mp

from ComplexToReal import ComplexToRealVec, RealToComplexVec, ComplextoRealMat, ReformatRealMatrix, MaxAbsComplexArray, \
    SumSqrAbsComplexArray
from SortedArrayIter import SortedArrayIter
from VectorOperations import NullSpace5, HodgeDual, E3, randomLoop, GradEqFromMathematica, mdot
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
    # NS = NullSpace4(ff(0, 10), ff(1, 10), ff(2, 10), ff(3, 10))
    NS = NullSpace5(np.array([ff(0, 10), ff(1, 10), ff(2, 10), ff(3, 10), ff(4, 10)], dtype=complex))
    pass


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
        # r = scipy.optimize.fsolve(reqs2, r, xtol=1e-14)
        # x = np.zeros(len(X0), dtype=complex)
        # RealToComplexVec(r, x)
        # err2 = SumSqrAbsComplexArray(Eqs2(x))
        # print("sum sqr error in equations further reduced from ", err1, " to ", err2)
    P = pack([x])
    return P


def stable_sum_real(a):
    if len(a) < 5:
        return math.fsum(a)
    absa = np.abs(a)
    ord = np.argsort(absa)[::-1]
    b = a[ord]
    c = b / b[0]
    if len(c) % 2 > 0:
        c = np.append(c, 0)
    c = c.reshape(-1, 2)
    c1 = np.apply_along_axis(np.sum, 1, c)
    return b[0] * stable_sum_real(c1)

def stable_sum(A):
    if A.dtype == complex:
        rr = A.real
        ii = A.imag
        return stable_sum_real(rr) + 1j * stable_sum_real(ii)
    else:
        return stable_sum_real(A)


def test_stable_sum():
    inputList = [1.23e+18, 1, -1.23e+18]
    # adding the sum of elements of the list using the fsum() function
    test = math.fsum(inputList)
    x = 5.
    test1 = math.exp(x) * stable_sum(np.array([(-x) ** n / math.factorial(n) for n in range(128)], dtype=float))
    y = 10. + 1j
    A = np.array([(-y) ** n / math.factorial(n) for n in range(40)], dtype=complex)
    test2 = np.exp(y) * stable_sum(A)
    pass


def numpy_combinations(x):# only different ones
    idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)
    return x[idx]

def test_all_combinations():
    N = 3
    test = numpy_combinations(np.arange(N))
    #np.array(np.meshgrid(array_1, array_2)).T.reshape(-1, 2)
    NN = np.array(np.meshgrid(np.arange(N),np.arange(N))).T.reshape(-1, 2)
    pass

def LogExpansion(f):
    # f(x) =  sum_0^M  f_k x^k + \dots
    # f_0 =1
    # g(x) = log(f(x)) = sum_1^{M} g_k x^k + \dots
    # f(x) g'(x) = f'(x)
    # \sum_{l=0}^{k} f_l (k-l) g_{k-l} = k f_k
    # k g_{k} = k f_{k} -\sum_{l=1}^{k} f_l (k-l) g_{k-l}
    M = len(f)
    g = [mpm.mpc(0) for i in range(M)]
    g[1] = f[1]
    for k in range(2, M):
        lst = [(k - l) * g[k - l] * f[l] for l in range(1, k)]
        g[k] = f[k] - mpm.fsum(lst) / k
    return g


def test_LogExpansion():
    f0 = [1. / mpm.factorial(n) for n in range(10)]
    g = LogExpansion(f0)
    assert (g[1] == mpm.mpc(1))
    lst = [x == mpm.mpc(0) for x in g[2:]]
    print(lst)

def ContinuedFractionCeffs(f):
    depth = len(f)
    a = f[0]
    if depth ==1:
        return [a]
    g = [1/f[1]]
    for l in range(2,depth):
        lst = [g[k]* f[l-k] for k in range(0, l-1)]
        test = mpm.fsum(lst)
        g.append(-test/f[1])
        pass
    ans =  ContinuedFractionCeffs(g)
    ans.append(a)
    return ans


def ValueOfContinuedFraction(coeffs, x):
    L = len(coeffs)
    ans = coeffs[0]
    for k in range(1,L):
        ans = coeffs[k] + x/ans
    pass
    return ans

def test_ContinuedFraction():
    import fractions  # Available in Py2.6 and Py3.0
    def approx2(c, maxd):
        'Fast way using continued fractions'
        return fractions.Fraction.from_float(c).limit_denominator(maxd)
    #1 - x/3 - x^2/45 - (2 x^3)/945 - x^4/4725 - (2 x^5)/93555
    mpm.dps = 30
    f0 = [mpm.mpf(x) for x in [1.,-1./3, -1./45, -2./945, -1./4725, - 2./93555]]
    coeffs = ContinuedFractionCeffs(f0)
    nicefrac = [ approx2(float(x),20) for x in coeffs]
    print(nicefrac)
    # [-11, 9,-7, 5,-3,1]
    value = ValueOfContinuedFraction(coeffs,mpm.mpf(1))
    test = 1/mpm.tan(mpm.mpf(1))
    pass


# & & W(\hat
# R) = \sum_
# {n_1
# n_2
# n_3
# n_4}  \frac
# {\prod_
# {k = 1} ^ 4(\I
# R_k) ^ {n_k} \binom
# {-\frac
# {1}
# {2}}{n_k}}{\Gamma(2 + \sum_1 ^ 4
# n_k)}
# a + x/b + x/c +x/d
class GroupFourierIntegral:
    def getFourIds(self, pair):
        N = self.N
        N1, N2 = pair
        n1, n2 = (int(N1 / N), N1 % N)
        n3, n4 = (int(N2 / N), N2 % N)
        return [n1, n2, n3, n4]

    def MakePairs(self):
        N =  self.N
        NN = np.arange(N * N, dtype=int)
        all_comb = np.array(np.meshgrid(NN,NN)).T.reshape(-1, 2)

        def degree(pair):
            return np.sum(self.getFourIds(pair))

        degrees = np.apply_along_axis(degree, 1, all_comb)
        ord = np.argsort(degrees)
        sorted_pairs = all_comb[ord]
        degrees = degrees[ord]
        good = degrees < N
        degrees = degrees[good]
        ranges = list(SortedArrayIter(degrees))
        self.degree_ranges = np.array(ranges, dtype=int)
        self.good_pairs = sorted_pairs[good]
        self.degree_ranges.tofile(os.path.join("plots", "pairs_degree_ranges." + str(self.N) + ".np"))
        self.good_pairs.tofile(os.path.join("plots", "group_integral_pairs." + str(self.N) + ".np"))

    def GetPairs(self):
        try:
            self.degree_ranges = np.fromfile(os.path.join("plots", "pairs_degree_ranges." + str(self.N) + ".np"),
                                             dtype=int).reshape(-1, 2)
            self.good_pairs = np.fromfile(os.path.join("plots", "group_integral_pairs." + str(self.N) + ".np"),
                                          dtype=int).reshape(-1, 2)
        except:
            self.MakePairs()

    def __init__(self, N):
        self.N = N
        self.GetPairs()
        s0 = np.matrix([[1, 0], [0, 1]])
        s1 = np.matrix([[0, 1], [1, 0]])
        s2 = np.matrix([[0, -1j], [1j, 0]])
        s3 = np.matrix([[1, 0], [0, -1]])
        self.Sigma = [s1, s2, s3]  # Pauli matrices
        self.Tau = [s0, 1j * s1, 1j * s2, 1j * s3]  # quaternions

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

    # symmegtrized as needed,  by symmetry
    def W(self, X):
        mpm.dps = 50
        R0 = self.GetRMatrix(X)
        R = mpm.matrix([[R0[i,j] for i in range(4)] for j in range(4)]) # symmetric 4 by 4 matricx out of general 3 by 3 matrix

        rr = mpm.eig(R)[0]
        TR = (mpm.fabs(rr[0]) + mpm.fabs(rr[1]) +mpm.fabs(rr[2]) +mpm.fabs(rr[3]))/4

        rrn = [[mpm.mpc( r / TR) ** k for k in range(self.N)] for r in rr]
        cc = [ [x[k] * mpm.binomial(mpm.mpf(-1./2),k) for k in range(self.N) ] for x in rrn]

        # test = np.product([cc[0,7],cc[1,4], cc[2,3]])
        def getTerm(pair):
            nn = self.getFourIds(pair)
            return mpm.mpc(cc[0] [nn[0]]) *mpm.mpc(cc[1][nn[1]]) *mpm.mpc(cc[2][nn[2]]) *mpm.mpc(cc[3][nn[3]])

        def SumRangeTerms(degree):
            range = self.degree_ranges[degree]
            beg, end = range
            return mpm.fsum(
                           [x for x in np.apply_along_axis(getTerm, 1, self.good_pairs[beg:end])]
                           )/ mpm.factorial(1 + degree)

        test = np.apply_along_axis(getTerm, 1, self.good_pairs[1:5])
        A = parallel_map(SumRangeTerms,range(len(self.degree_ranges)),0)

        factor = A[0]
        A = [x/factor for x in A]
        Lim = len(A) - len(A)%2
        a = LogExpansion(A[:Lim])
        coeffs = ContinuedFractionCeffs(a)
        ans0 = ValueOfContinuedFraction(coeffs,     mpm.mpc(-1j *TR))
        ans2 = ValueOfContinuedFraction(coeffs[2:], mpm.mpc(-1j *TR))
        assert mpm.fabs(ans0 - ans2) < 1e-2
        # test = np.log(stable_sum(A)) -complex(ValueOfContinuedFraction(coeffs,     mpm.mpc(1)))
        # test1 = ValueOfContinuedFraction(coeffs,     mpm.mpc(1))-ValueOfContinuedFraction(coeffs[2:], mpm.mpc(1))
        return factor * complex(mpm.exp(ans0))
        pass


def test_GroupFourierIntegral():
    N = 40
    gfi = GroupFourierIntegral(N)
    r = np.random.normal(scale=0.1, size=30) + 1j * np.random.normal(scale=0.1, size=30)
    r = r.reshape((3, 10))
    X = r.dot(r.T)
    test = gfi.W(X)
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

    def SaveCurve(self, cycle, node_num):
        self.Frun.tofile(self.GetSaveFilename(cycle, node_num))

    def CollectStatistics(self, C0, t0, t1, time_steps):
        CDir = 0.5 * (C0 + np.roll(C0, 1, axis=0))
        CRev = CDir[::-1]
        M = self.M
        mpm.dps = 40
        pathnames = []
        GFI = GroupFourierIntegral(40)
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

        factor = 1e6

        def Psi(pathname):
            ans = []
            try:
                F = np.fromfile(pathname, dtype=complex)
                F = F.reshape((-1, 3))
                assert F.shape == (M, 3)  # (M by 3 complex tensor)
                Q = np.roll(F, 1, axis=0) - F  # (M by 3 complex tensor)
                X1 = np.dot(CDir.T, Q)  # (3 by 3 complex tensor for direct curve)
                X2 = np.dot(CRev.T, np.conjugate(Q))  # (3 by 3 complex tensor for reflected curve)
                for tt in np.linspace(t0, t1, time_steps):
                    t = tt * factor
                    psi = 0. + 0.j
                    for X in (X1 / np.sqrt(t), X2 / np.sqrt(t)):
                        psi += GFI.W(X)
                    pass
                    ans.append([t, psi])
                return ans
            except Exception as ex:
                print(ex)
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
        psidata = psidata.transpose((2, 1, 0))
        times = np.mean(psidata[0], axis=1)  # (time_steps,N)->time_steps
        psiR = np.mean(psidata[1].real, axis=1)  # (time_steps,N)->time_steps
        psiI = np.mean(psidata[1].imag, axis=1)  # (time_steps,N)->time_steps

        XYPlot([psiR, psiI], plotpath=os.path.join(self.GetSaveDirname(), "WilsonLoop.png"),
               scatter=True,
               title='Wilson Loop')


def runIterMoves(num_vertices=100, num_cycles=10, T=1.0, num_steps=1000,
                 t0=1., t1=10., time_steps=100,
                 node=0, NewRandomWalk=False, plot=True):
    M = num_vertices
    mover = IterMoves(M)
    # print_debug("created mover(",M,")")
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
