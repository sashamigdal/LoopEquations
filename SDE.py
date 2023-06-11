import numpy as np
import os, sys
from numpy import cos, sin, pi
from scipy.linalg import pinvh, null_space
import sdeint
import multiprocessing as mp
from parallel import parallel_map, ConstSharedArray
from plot import Plot, PlotTimed, MakeDir

MakeDir("plots")


# no changes, git test2

def test_ItoProcess():
    A = np.array([[-0.5, -2.0],
                  [2.0, -1.0]])

    B = np.diag([0.5, 0.5])  # diagonal, so independent driving Wiener processes

    tspan = np.linspace(0.0, 10.0, 100000)
    x0 = np.array([3.0, 3.0])

    def f(x, t):
        return A.dot(x)

    def G(x, t):
        return B

    result = sdeint.itoSRI2(f, G, x0, tspan).reshape(100, -1, 2)

    shared = [[ConstSharedArray(x)] for x in result[10:]]
    print("finished Ito process")
    mp.set_start_method('fork')

    def Mean(x):
        return np.mean(x[:])

    cores = mp.cpu_count()
    means = parallel_map(Mean, shared, cores)
    Plot(means, os.path.join("plots", "test_ito_euler.png"), x_label='t', y_label='m', title='Ito')


def ff(k, M):
    delta = pi / M
    return np.array([cos(2 * k * delta), sin(2 * k * delta), cos(delta) * 1j]) / (2 * sin(delta))


def RI(X):
    return np.array([X.real, X.imag]).transpose()


def gamma_formula(u, ans):
    (N, d) = u.shape
    V = np.array([u.real, u.imag]).reshape(2, -1)
    Q = pinvh(V.dot(V.T))
    np.dot(Q, V, V)
    V1 = V.reshape(2, N, d)
    ans[:] = V1[1] + V1[0] * 1j


def HodgeDual(v):
    return np.array([
        [0, v[2], -v[1]],
        [-v[2], 0, v[0]],
        [v[1], -v[0], 0]
    ], dtype=v.dtype)


E3 = np.array([HodgeDual(v) for v in np.eye(3)], dtype=float)


def test_LevyCivita():
    a = np.array([1, 2, 3], float)
    b = np.array([4, 5, 6], float)
    ad = a.dot(E3)
    c = ad.dot(b)
    c1 = np.cross(a, b)
    test = c - c1
    test


# M = Length[F];
# IM = ConstantArray[I,{M}];
# q = Thread[RotateRight[F,1]- F,1];
# (*Return[MatrixForm[q]];*)
# qd = Normal @ HodgeDual[#] &/@ q;
# (*Return[MatrixForm[qd]];*)
# G =2 F[[#]]q[[#]]-I &/@ Range[M];
# (*Return[G];*)
# (* G is a complex M-array *);
# U =-G[[#]] F[[#]]. qd[[#]] &/@ Range[M];
# (* U is a  M-array of complex 3 vectors*);
# (*Return[U];*)
# \[Gamma] =\[Gamma]formula/@ U;
# (*If[Length[Flatten[\[Gamma]null/@ U]] >0, Print["Nullspace!"]];*)
# (*Return[\[Gamma]];*)
# U =-G[[#]] F[[#]]. qd[[#]] &/@ Range[M];
# V =2 G[[#]]*q[[#]]-F[[#]] &/@ Range[M];
# (* V is a  M-array of complex 3 vectors*);
# (*Return[V];*)
# (* the arrays  of matrices *)
# Z3 =ConstantArray[0,{3,3}];
# LL = MyTable[If[j< i,\[Gamma][[i]]\[TensorProduct]V[[i]].qd[[j]],Z3],{i,M},{j,M}];
# X = Re[LL];
# X[[#,#]] += Re[\[Gamma][[#]]\[TensorProduct]U[[#]]]& /@ Range[M];
# (*Print["X:", Dimensions[X]];
# Print["LL:", Dimensions[LL]];*)
# (*Return [X];*)
# (*Print["MDot[Im[LL],X]:"];
# Return[MDot[Im[LL],X]];*)
# MM = X;
# (*Print["MM:", Dimensions[MM]];*)
# Do[MM = X - MDot[Im[LL],MM],M-1];
# (* The arrays of vectors *)
# (*Return[MM];*)
# Y =\[Gamma][[#]]\[TensorProduct]V[[#]] & /@ Range[M];
# (*M array of 3 X 3 complex matrix *)
# (*Return[Y];*)
# PP = Re[Y];
# QQ = -Im[Y];
# Do[PP =Re[Y] +MVDot[ Im[LL],PP],M-1];
# Do[QQ =-Im[Y] +MVDot[ Im[LL],QQ],M-1];
# P= VDot[qd,PP];
# (*3 X 3 complex matrix *)
# Q=VDot[qd,PP];
# (*3 X 3 complex matrix *)
# {PQM,NS} =PseudoInverseFromBlock[{ {Re[P],Re[Q]}, {Im[P], Im[Q]}}];
# (* PQM is 2 X 2 block matrix with 3X3 real matrix elements *)
# NS = (#[[1]]- I #[[2]])&/@NS;
# K = Length[NS];
# (*Return[NS];*)
# I3 = IdentityMatrix[3];
# X = VDot[VMDot[{I3, I I3}, PQM], {I3, - I I3}];
# (* X is a 3X3 complex matrix *)
# (*Return[X];*)
# (*Return[NS];*)
# (* Y is a M vector with elements being complex 3X3 matrices*)
# Y = Re[X . #] & /@ qd;
# (* MM is a M by M matrix with elements being real 3X3 matrices*)
# (* Lambda is a M vector with elements being 3X3 complex matrices*)
# \[CapitalLambda] = -(Im[ X . #] & /@ qd) - VMDot[Y, MM];
# (*Return[\[CapitalLambda]];*)
# (* Z is a K by M matrix with elements being complex 3 vectors*)
# Z = MyTable[ NS[[i]] . qd[[n]], {i, K}, {n, M}];
# (*Return[Z];*)
# (* Theta is a K by M matrix with elements being real 3 vectors*)
# \[CapitalTheta] = Im[Z] - MDot[Re[Z], MM];
# (*Return[\[CapitalTheta]];*)
# CC = MDot[\[CapitalTheta], Transpose[\[CapitalTheta]]];
# (*Return[MatrixForm[CC]];*)

class SDEProcess():
    global M, M3, LL, C33, R, C, RR, CC

    def __init__(self):
        pass

    def Matrix(self, F):
        q = np.roll(F, 1, axis=0) - F
        # qd0 = np.array([HodgeDual(v) for v in q])
        qd = C33[0]
        np.dot(q, E3, qd)
        G = C[0]
        for k in range(M): G[k] = 2 * F[k].dot(q[k]) - 1j
        U = C[1]
        for k in range(M): U[k] = -G[k] * F[k].dot(qd[k])
        gamma = C[2]
        gamma_formula(U, gamma)
        V = C[3]
        for k in range(M): V[k] = 2 * G[k] * q[k] - F[k]
        gammaXV = C33[1]
        for k in range(M): gammaXV[k] = np.kron(gamma[k], V[k]).reshape(3, 3)
        LL[:] = gammaXV.reshape(M3, 3).dot(qd.reshape(M3, 3).T).reshape(M, 3, M, 3).transpose((0, 2, 1, 3))
        XX = RR[0]
        XX[:] = LL.real
        for k in range(M):
            XX[k, k] -= np.kron(gamma[k], U[k]).real.reshape(3, 3)
        XX = XX.transpose((0, 2, 1, 3)).reshape(M3, M3)
        MM = RR[1].transpose((0, 2, 1, 3)).reshape(M3, M3)
        MM[:] = XX
        LLI = RR[2].transpose((0, 2, 1, 3)).reshape(M3, M3)
        LLI[:] = -LL.imag.transpose((0, 2, 1, 3)).reshape(M3, M3)
        for _ in range(M - 1):
            np.dot(LLI, MM, MM)
            MM[:] += XX
        pass
        Y = C33[2]
        for k in range(M):
            Y[k] = np.kron(gamma[k], V[k]).reshape(3, 3)
        Y = Y.reshape(M3, 3)
        PP = Y.real
        QQ = -Y.imag
        for _ in range(M - 1):
            np.dot(LLI, PP, PP)
            np.dot(LLI, QQ, QQ)
            PP[:] += Y.real
            QQ[:] -= Y.imag
        pass
        QD = qd.transpose((1, 0, 2)).reshape(3, M3)
        P = np.dot(QD, PP)
        Q = np.dot(QD, QQ)
        BB = np.array([[P.real, Q.real], [P.imag, Q.imag]]).transpose(0, 2, 1, 3).reshape(6, 6)
        NS = null_space(BB).reshape(2, 3, -1)
        NS = (NS[0] - 1j * NS[1]).transpose()
        BB = pinvh(BB)
        K = len(NS)
        PQM = BB.reshape(2, 3, 2, 3).transpose((0, 2, 1, 3))
        X = PQM[0, 0] + PQM[1, 1] - 1j * PQM[1, 0] + 1j * PQM[0, 1]
        Z = X.dot(qd.transpose(1, 2, 0).reshape(3, 3 * M))
        Lambda = -Z.imag - Y.T.dot(MM)
        Z = NS.dot(QD)
        Theta = Z.imag - Z.real.dot(MM)
        CC = Theta.dot(Theta.T)
        # eig=np.linalg.eigh(CC)
        CCI = np.linalg.inv(CC)
        Y = CCI.dot(Theta)
        Lambda -= Lambda.dot(Theta.T).dot(Y)
        # test =Lambda.dot(Theta.T)
        # test
        TT = MM - MM.dot(Theta.T).dot(Y)
        TT = TT.astype(complex).reshape(M, 3, M, 3).transpose((0, 2, 1, 3))
        TT[0] += Lambda.reshape(3, 3, M).transpose((2, 0, 1))
        for k in range(M - 1):
            TT[k + 1] += TT[k]
        return TT.transpose((0, 2, 1, 3)).reshape(3 * M, 3 * M)

    def ItoProcess(self, T, num_steps, chunk):
        tspan = np.linspace(0.0, T, num_steps)

        def f(F, t):
            return self.Matrix(F)

        def G(x, t):
            return np.zeros(self.M, dtype=complex)

        result = sdeint.itoSRI2(f, G, F0, tspan).reshape(chunk, -1, len(F0))

        shared = [[ConstSharedArray(x)] for x in result[10:]]
        print("finished Ito process")
        mp.set_start_method('fork')

        def Mean(x):
            return np.mean(x[:])

        cores = mp.cpu_count()
        means = parallel_map(Mean, shared, cores)
        Plot(means, os.path.join("plots", "test_ito_euler.png"), x_label='t', y_label='m', title='Ito')


############### tests
def testSDE():
    global M, M3, LL, C33, R, C, RR, CC
    M = 4
    M3 = M * 3
    F0 = np.array([ff(k, M) for k in range(M)], complex)
    R = [np.zeros((M, 3), float) for _ in range(3)]
    C = [np.zeros((M, 3), complex) for _ in range(4)]
    RR = [np.zeros((M, M, 3, 3), float) for _ in range(3)]
    CC = [np.zeros((M, M, 3, 3), complex) for _ in range(3)]
    C33 = [np.zeros((M, 3, 3), complex) for _ in range(2)]
    LL = CC[0]

    SD = SDEProcess()
    SD.Matrix(F0)


def testFF():
    print(ff(3, 10))


def testRI():
    print(RI(np.array([1 + 2j, -1 + 3j, -5 + 6j])))


def test_gamma_formula():
    u = np.array(
        [[1 + 1j, 2 + 2j, 3 + 3j], [1 - 1j, 2 - 2j, 3 - 3j], [5 - 6j, 7 - 8j, 9 - 10j], [-5 - 6j, -7 - 8j, -9 - 10j]])
    gamma = np.zeros_like(u)
    gamma_formula(u, gamma)
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
    testFF()
