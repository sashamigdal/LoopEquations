import concurrent.futures
from os.path import split

import numpy as np
import os, sys
from numpy import cos, sin, pi
from scipy.linalg import pinvh, null_space
from scipy.stats import ortho_group
from mpmath import hyp0f1, hyp1f2
import sdeint
import multiprocessing as mp
# from parallel import parallel_map, ConstSharedArray, WritableSharedArray
from plot import Plot, PlotTimed, MakeDir, XYPlot
from Timer import MTimer as Timer

MakeDir("plots")


def ff(k, M):
    delta = pi / M
    return np.array([cos(2 * k * delta), sin(2 * k * delta), cos(delta) * 1j]) / (2 * sin(delta))


def cc(k, M):
    delta = pi / M
    return np.array([cos(2 * k * delta), sin(2 * k * delta), 0], dtype=float)


def RI(X):
    return np.array([X.real, X.imag]).transpose()


def testDot3():
    a = np.random.random(4).reshape(2, 2)
    b = np.random.random(4).reshape(2, 2)
    c = np.zeros((2, 2), dtype=a.dtype)
    np.dot(a, b, c)
    np.dot(a, b, a)
    assert (a == c).all()


def gamma_formula(u):
    assert u.ndim == 1
    d = u.shape[0]
    V = np.vstack([u.real, u.imag]).reshape(2, d)
    Q = pinvh(V.T.dot(V))
    V1 = np.dot(Q, V.T)  # (d,2)
    return V1[:, 1] + V1[:, 0] * 1j


def VecGamma(U, Ans):
    Ans[:] = np.vstack([gamma_formula(u) for u in U])


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


def dot_shape(a, b):
    x = np.zeros_like(a)
    y = np.zeros_like(b)
    return x.dot(y).shape


def VecKron(a, b):
    M = a.shape[0]
    assert (M == b.shape[0])
    S = a.shape[1:]
    T = b.shape[1:]
    MST = (M, *S, *T)
    return np.vstack(np.kron(a[k], b[k]) for k in range(M)).reshape(MST)


def VecKron22(A, B, C):
    assert A.ndim == 2
    assert B.ndim == 2
    C[:] = A[:, :, None] * B[:, None, :]


def parKron22(A, B, C):
    assert B.shape == A.shape
    assert A.ndim == 2
    N, d = A.shape
    assert C.shape == (N, d, d)

    def func(i):
        C[i] = np.kron(A[i, :], B[i, :]).reshape(d, d)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(func, range(N))

    pass


def VecDot(a, b):
    M = a.shape[0]
    assert (M == b.shape[0])
    x = np.vstack(a[k].dot(b[k]) for k in range(M))
    ST = dot_shape(a[0], b[0])
    MST = (M, *ST)
    return x.reshape(MST)


def SetDiag(X, f):
    N = len(f)
    assert X.shape[:2] == (N, N)

    def func(k):
        X[k, k] = f[k]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(func, range(N))


def ZeroBelowEqDiag(X):
    assert X.flags.c_contiguous
    N = len(X)
    assert X.shape[:2] == (N, N)

    def func(k):
        X[k, k:] = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(func, range(N))


def testVecOps():
    mp.set_start_method('fork')
    a = np.arange(12).reshape(4, 3)
    b = np.arange(10, 22).reshape(4, 3)
    c = np.zeros((4, 3, 3))
    parKron22(a, b, c)
    d = VecDot(a, b)
    assert d.ndim == 1
    assert d.shape[0] == 4
    X1 = np.arange(36).reshape(3, 3, 2, 2)
    ZeroBelowEqDiag(X1)


class SDEProcess():
    global M, M3, LL, C33, R, C, RR, CC, G

    def __init__(self):
        pass

    def Matrix(self, F_flattened):
        F = F_flattened.reshape(M, 3)
        q = np.roll(F, 1, axis=0) - F
        # qd0 = np.array([HodgeDual(v) for v in q])
        qd = C33[0]
        np.dot(q, E3, qd)
        FXq = C33[1]
        parKron22(F, q, FXq)
        G[:] = np.trace(FXq, axis1=1, axis2=2)
        G[:] = 2 * G - 1j  # OK, checked  with the paper
        U = C[1]
        U[:] = np.trace(FXq.dot(E3), axis1=1, axis2=2)
        U[:] = G.reshape(M, 1) * U  # OK, checked  with the paper
        # for k in range(M): U[k] = -G[k] * F[k].dot(qd[k])
        gamma = C[2]
        VecGamma(U, gamma)  # OK, fixed bug  with the paper
        V = C[3]
        V[:] = G.reshape(M, 1) * q - 2 * F  # OK, fixed bug  with the paper
        gammaXV = C33[1]
        parKron22(gamma, V, gammaXV)  # OK, fixed bug  with the paper
        LL[:] = -gammaXV.reshape(M3, 3).dot(qd.reshape(M3, 3).T).reshape(M, 3, M, 3).transpose((0, 2, 1, 3))
        # LL = gamma_k X Vk dot hat q
        ZeroBelowEqDiag(LL)
        XX = RR[0]
        XX[:] = LL.real  # Mk>l  = Re( gamma_k X Vk dot hat q)      # Mk<l =0
        gammaXU = C33[4]
        parKron22(gamma, U, gammaXU)  # OK, checked  with the paper
        SetDiag(XX, gammaXU.real)  # Mkk  = Re( gamma_k X U_k)
        XX = XX.transpose((0, 2, 1, 3)).reshape(M3, M3)
        MM = RR[1].transpose((0, 2, 1, 3)).reshape(M3, M3)
        MM[:] = XX  # eq 2.135b wihout last term
        LLI = RR[2].transpose((0, 2, 1, 3)).reshape(M3, M3)
        LLI[:] = -LL.imag.transpose((0, 2, 1, 3)).reshape(M3, M3)
        TMP = RR[3].reshape(M3, M3)
        for _ in range(M - 1):
            np.dot(LLI, MM, TMP)
            TMP += XX
            MM[:] = TMP
        pass
        TMP = C33[3].reshape(M3, 3)
        Gamma = C33[4].reshape(M3, 3)
        Gamma[:] = gammaXV.reshape(M3, 3)
        LLI *= -1
        for _ in range(M - 1):
            np.dot(LLI, np.conjugate(Gamma), TMP)
            TMP[:] += gammaXV.reshape(M3, 3)
            Gamma[:] = TMP
        PP = Gamma.real
        QQ = -Gamma.imag
        QD = qd.transpose((1, 0, 2)).reshape(3, M3)
        P = np.dot(QD, PP)
        Q = np.dot(QD, QQ)
        H = np.array([[P.real, Q.real], [P.imag, Q.imag]]).transpose(0, 2, 1, 3).reshape(6, 6)
        NS = null_space(H).reshape(2, 3, -1)
        NS = (NS[0] - 1j * NS[1]).transpose()
        K = len(NS)
        PQM = pinvh(H).reshape(2, 3, 2, 3).transpose((0, 2, 1, 3))
        X = PQM[0, 0] + PQM[1, 1] - 1j * PQM[1, 0] + 1j * PQM[0, 1]
        Q = C33[5].transpose(1, 0, 2).reshape(3, M3)
        Q[:] = qd.transpose(1, 0, 2).reshape(3, M3)
        R = C33[6].transpose(1, 0, 2).reshape(3, M3)
        np.dot(Q, MM, R)
        R[:] = 1j * Q - R
        Lambda = X.dot(R)
        # HERE I STOPPED
        Theta = np.conjugate(NS).dot(R).real
        cc = Theta.dot(Theta.T)
        eig = np.linalg.eigh(cc)
        cci = pinvh(cc)
        Y = cci.dot(Theta)
        Lambda -= Lambda.dot(Theta.T).dot(Y)
        # test =Lambda.dot(Theta.T)
        # test
        GL = CC[4].reshape(M3, M3)
        np.dot(Gamma, Lambda, GL)
        MM += GL.real
        TMP1 = RR[3].reshape(M3, M3)
        TMP1[:] = MM - MM.dot(Theta.T).dot(Y)
        TT = CC[4]
        TT[:] = TMP1.astype(complex).reshape(M, 3, M, 3).transpose((0, 2, 1, 3))
        TT[0] += Lambda.reshape(3, 3, M).transpose((2, 0, 1))
        for k in range(M - 1):
            TT[k + 1] += TT[k]
        return TT.transpose((0, 2, 1, 3)).reshape(M3, M3)

    def ItoProcess(self, F0, T, num_steps, chunk, node_num):
        tspan = np.linspace(0.0, T, num_steps)

        def f(x, t):
            return np.zeros((M3,), dtype=complex)

        def g(F, t):
            return self.Matrix(F)

        with Timer("ItoProcess with N=" + str(M) + " and " + str(num_steps) + " steps"):
            result = np.array(sdeint.itoEuler(f, g, F0, tspan)).reshape(-1, M3)

        result.tofile(os.path.join("plots", "test_ito_euler." + str(node_num) + ".np"))

    def PlotResults(self, C0):
        C1 = C0 + np.roll(C0, 1, axis=0)
        C1 = 0.5 * C1

        def Psi(R):
            F = R.reshape(M, 3)  # (M by 3 complex tensor)
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

        result = None
        for filename in os.listdir("plots"):
            if filename.endswith(".np"):
                try:
                    node = int(filename.split(".")[-2])
                    data = np.fromfile(os.path.join("plots", "test_ito_euler." + str(node) + ".np"),
                                       dtype=complex).reshape(-1, M3)
                    result = data if (result is None) else np.append(result, data)
                except:
                    print("could not read ", filename)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            emap = executor.map(Psi, result)

        psidata = np.array(list(emap), complex)
        XYPlot([psidata.real, psidata.imag], plotpath=os.path.join("plots", "test_ito_euler.png"), scatter=True,
               title='Ito')


############### tests
def runSDE(number_of_vertices, node_num=0):
    global M, M3, LL, C33, R, C, RR, CC, G
    # mp.set_start_method('fork')
    M = number_of_vertices
    M3 = M * 3
    np.random.seed(node_num)
    F0 = np.array([ff(k, M) for k in range(M)], complex)
    m = ortho_group.rvs(dim=3)
    F0 = F0.dot(m).reshape(M3)
    R = [np.zeros((M, 3), float) for _ in range(6)]
    C = [np.zeros((M, 3), complex) for _ in range(6)]
    RR = [np.zeros((M, M, 3, 3), float) for _ in range(6)]
    CC = [np.zeros((M, M, 3, 3), complex) for _ in range(6)]
    C33 = [np.zeros((M, 3, 3), complex) for _ in range(10)]
    LL = CC[0]
    G = np.zeros((M,), complex)

    SD = SDEProcess()
    B = SD.Matrix(F0)
    SD.ItoProcess(F0, 0.1, 100, 2, node_num)


def NullSpace3(F0, F1, F2):
    q0 = F1 - F0
    q1 = F2 - F1
    # dq0 = dt0 X q0 = dt0.dot.E3.dot.q0
    # dq1 = dt1 X q1 = dt1.dot.E3.dot.q1
    # dt0 .dot q0 X F0 = F0 dot dt0 X q0 =0
    # dt1 .dot q1 X F2 = F2 dot dt1 X q1 =0
    # dq0 + dq1 =0
    Z = np.zeros((3), dtype=complex)
    Z2 = np.zeros((3,3), dtype=complex)
    lst = [[q0.dot(E3.dot(F0)), Z], [Z, q1.dot(E3.dot(F2))]]
    A = np.array(lst).reshape(2, 6)
    B = np.array([E3.dot(q0), E3.dot(q1)]).transpose(1, 0, 2).reshape(3, 6)
    X = np.vstack([A, B])  # 5 X6
    # X.dot(np.array([dt0,dt1]).reshape(6))  =0
    NS = null_space(X)
    test0 = X.dot(NS)
    Q1 = np.array([E3.dot(q0),Z2]).transpose(1,0,2).reshape(3,6)
    test1 = Q1.dot(NS)
    Q2 = np.array([Z2,E3.dot(q1)]).transpose(1, 0, 2).reshape(3, 6)
    test = Q2.dot(NS)
    Q1 = np.hstack([qd0, qd1])
    return NS.T.reshape(-1, 2, 3)


def NullSpace4(F0, F1, F2, F3):
    q0 = F1 - F0
    q1 = F2 - F1
    q2 = F3 - F2
    #############################
    # (F0 dot q0 - i/2)^2 = F0^2 + (2 i -1)/4  # scalar equation, (trivial, as delta F0 =0)  #0
    # (F3 dot q2 + i/2)^2 = F3^2 + (2 i -1)/4 # scalar equation (trivial, as delta F3 =0)   #1


    # qd_k = E3 dot q_k
    # G1 = 2 F1 dot q1 - i # scalar equation
    # dq_k = qd_k dot dt_k
    #############################
    # (F0 dot qd0) dot dt0  + 0 dot dt1 + 0 dot dt2= 0  # scalar eq 0
    # 0 dot dt0 + 0 dot dt1 + (F3 dot qd2) dot dt2 = 0  # scalar eq 1

    # (F1 dot q1 - i/2)^2 = F1^2 + (2 i -1)/4  # scalar equation # scalar eq 2
    # G1 F1 dot dq1 +  G1 q1 dot d F1 - 2 F1 dot dF1 =0 # scalar eq 2
    # (G1 q1 dot qd0 - 2 F1 dot qd0) dot dt0   + (G1 F1 dot qd1) dot dt1  + 0 dot dt2 =0  # scalar eq 2

    # qd0 dot dt0 + qd1 dot dt1 + qd2 dot dt2 =0 # vector eq 1
    #
    Z = np.zeros((3), dtype=complex)
    Z2 = np.zeros((3,3), dtype=complex)
    qd0 = E3.dot(q0)
    qd1 = E3.dot(q1)
    qd2 = E3.dot(q2)
    G1 = 2 * F1.dot(q1) - 1j
    lst = []
    lst.append([F0.dot(qd0), Z, Z])  # scalar eq 0
    lst.append([Z, Z, F3.dot(qd2)])  # scalar eq 1
    lst.append([G1 * q1.dot(qd0) - 2 * F1.dot(qd0), G1 * F1.dot(qd1), Z])  # scalar eq 2
    A = np.array(lst).reshape(3, 9)  # 3e,3t, d ->3e, 3 *d
    B = np.array([qd0, qd1, qd2]).transpose(1, 0, 2).reshape(3, 9)  # (3t,d,d) ->  (d, 3t*d)
    X = np.vstack([A, B])  # (3 + d)) X 3 d
    NS = null_space(X)
    test0 = X.dot(NS)
    Q0 = np.array([E3.dot(q0), E3.dot(q1),E3.dot(q2)]).transpose(1,0,2).reshape(3,9)
    Q1 = np.array([Z2, Z2, E3.dot(q2)]).transpose(1,0,2).reshape(3,9)
    Q2 = np.array([Z2, E3.dot(q1), Z2]).transpose(1,0,2).reshape(3,9)
    Q3 = np.array([E3.dot(q0), Z2, Z2]).transpose(1,0,2).reshape(3,9)
    test = [Q0.dot(NS),Q1.dot(NS),Q2.dot(NS),Q3.dot(NS)]
    return NS


def testNullSpace():
    NS = NullSpace3(ff(0, 10), ff(1, 10), ff(2, 10))
    NS = NullSpace4(ff(0, 10), ff(1, 10), ff(2, 10), ff(3, 10))
    pass


class IterMoves():
    def __init__(self, M):
        self.M = M
        self.Fstart = np.array([ff(k, M) for k in range(M)], complex)
        self.Frun = self.Fstart.copy()

    def MoveOneVertex(self, k, T, num_steps):
        M = self.M
        F0 = self.Frun[(k + M - 1) % M]
        F1 = self.Frun[k]
        F2 = self.Frun[(k + 1) % M]

        Z = np.zeros_like(F1)
        tspan = np.linspace(0.0, T, num_steps)
        NS = None

        def f(x, t):
            return Z

        def g(q, t):
            NS = NullSpace(F0, F0 + q, F2)  # K, 2,3
            K = NS.shape[0]
            Y = NS.dot(E3).dot(q).transpose(1, 2, 0)  # K, 2,3 -> 2,3,K
            X = Y[0].reshape(3, K)
            return X

        noise = np.random.normal(size=2) + 1j * np.random.normal(size=2)
        dq = g(F1 - F0, 0).dot(noise)  # delta q = test.dot(q)
        test1 = dq.dot(F1 - F0)
        test2 = dq.dot(F0)
        dq
        with Timer("ItoProcess with N=" + str(M) + " and " + str(num_steps) + " steps"):
            result = sdeint.itoEuler(f, g, F1 - F0, tspan)
        return result[-1]


def test_IterMoves():
    Moves = IterMoves(10)
    Moves.MoveOneVertex(0, 1, 100)


def testSDE():
    M = 100
    runSDE(M, 0)
    C0 = np.array([cc(k, M) for k in range(M)])
    m = ortho_group.rvs(dim=3)
    C0 = C0.dot(m)
    SDEProcess().PlotResults(C0)


def testFF():
    print(ff(3, 10))


def testRI():
    print(RI(np.array([1 + 2j, -1 + 3j, -5 + 6j])))


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
        runSDE(N, P)
