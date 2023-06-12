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


def testDot3():
    a = np.random.random(4).reshape(2, 2)
    b = np.random.random(4).reshape(2, 2)
    c = np.zeros((2, 2), dtype=a.dtype)
    np.dot(a, b, c)
    np.dot(a, b, a)
    assert (a == c).all()


def gamma_formula(u, ans):
    assert u.ndim ==1
    d = u.shape[0]
    V = np.vstack([u.real, u.imag]).reshape(2, d)
    Q = pinvh(V.T.dot(V))
    V1 = np.dot(Q, V.T)# (d,2)
    ans[:] = V1[:,1] + V1[:,0] * 1j

def VecGamma(U, Ans):
    pass
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


def VecDot(a, b):
    M = a.shape[0]
    assert (M == b.shape[0])
    x = np.vstack(a[k].dot(b[k]) for k in range(M))
    ST = dot_shape(a[0], b[0])
    MST = (M, *ST)
    return x.reshape(MST)


def testVecOps():
    a = np.arange(12).reshape(4, 3)
    b = np.arange(10, 22).reshape(4, 3)
    c = VecKron22(a, b)
    assert c.shape == (4, 3, 3)
    d = VecDot(a, b)
    assert d.ndim == 1
    assert d.shape[0] == 4


class SDEProcess():
    global M, M3, LL, C33, R, C, RR, CC, G

    def __init__(self, F0):
        self.F0 = F0

    def Matrix(self, F):
        q = np.roll(F, 1, axis=0) - F
        # qd0 = np.array([HodgeDual(v) for v in q])
        qd = C33[0]
        np.dot(q, E3, qd)
        TMP = C33[1]
        VecKron22(F, q, TMP)
        G[:] = np.trace(TMP, axis1=1, axis2=2)
        G[:] = 2 * G - 1j           #OK, checked  with the paper
        U = C[1]
        U[:] = np.trace(TMP.dot(E3), axis1=1, axis2=2)
        U[:] = G.reshape(M, 1) * U               #OK, checked  with the paper
        # for k in range(M): U[k] = -G[k] * F[k].dot(qd[k])
        gamma = C[2]
        gamma_formula(U, gamma)        #OK, checked  with the paper
        V = C[3]
        V[:] = G.reshape(M, 1) * q - 2 * F        #OK, fixed bug  with the paper
        gammaXV = C33[1]
        VecKron22(gamma, V, gammaXV)
        LL[:] = gammaXV.reshape(M3, 3).dot(qd.reshape(M3, 3).T).reshape(M, 3, M, 3).transpose((0, 2, 1, 3))
        XX = RR[0]
        XX[:] = LL.real
        TMP = C33[4]
        VecKron22(gamma, U, TMP)
        for k in range(M):
            XX[k, k] -= TMP[k].real
        XX = XX.transpose((0, 2, 1, 3)).reshape(M3, M3)
        MM = RR[1].transpose((0, 2, 1, 3)).reshape(M3, M3)
        MM[:] = XX
        LLI = RR[2].transpose((0, 2, 1, 3)).reshape(M3, M3)
        LLI[:] = -LL.imag.transpose((0, 2, 1, 3)).reshape(M3, M3)
        TMP = RR[3].reshape(M3, M3)
        for _ in range(M - 1):
            np.dot(LLI, MM, TMP)
            TMP += XX
            MM[:] = TMP
        pass
        Y = C33[2]
        VecKron22(gamma, V, Y)
        Y = Y.reshape(M3, 3)
        TMP1 = C33[3].reshape(M3, 3)
        TMP2 = C33[4].reshape(M3, 3)
        TMP2[:] = Y
        for _ in range(M - 1):
            np.dot(LLI, np.conjugate(TMP2), TMP1)
            TMP1[:] += Y
            TMP2[:] = TMP1
        PP = TMP2.real
        QQ = - TMP2.imag
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
        cc = Theta.dot(Theta.T)
        # eig=np.linalg.eigh(CC)
        cci = np.linalg.inv(cc)
        Y = cci.dot(Theta)
        Lambda -= Lambda.dot(Theta.T).dot(Y)
        # test =Lambda.dot(Theta.T)
        # test
        TMP1 = RR[3].reshape(M3, M3)
        TMP1[:] = MM - MM.dot(Theta.T).dot(Y)
        TT = CC[4]
        TT[:] = TMP1.astype(complex).reshape(M, 3, M, 3).transpose((0, 2, 1, 3))
        TT[0] += Lambda.reshape(3, 3, M).transpose((2, 0, 1))
        for k in range(M - 1):
            TT[k + 1] += TT[k]
        return TT.transpose((0, 2, 1, 3)).reshape(M3, M3)

    def ItoProcess(self, T, num_steps, chunk):
        tspan = np.linspace(0.0, T, num_steps)

        def f(F, t):
            return self.Matrix(F)

        def G(x, t):
            return np.zeros(self.M, dtype=complex)

        result = sdeint.itoEuler(f, G, self.F0, tspan).reshape(chunk, -1, len(self.F0))

        shared = [[ConstSharedArray(x)] for x in result[1:]]
        print("finished Ito process")
        mp.set_start_method('fork')

        def Mean(x):
            return np.mean(x[:])

        cores = mp.cpu_count()
        means = parallel_map(Mean, shared, cores)
        Plot(means, os.path.join("plots", "test_ito_euler.png"), x_label='t', y_label='m', title='Ito')


############### tests
def testSDE():
    global M, M3, LL, C33, R, C, RR, CC, G
    M = 4
    M3 = M * 3
    F0 = np.array([ff(k, M) for k in range(M)], complex)
    R = [np.zeros((M, 3), float) for _ in range(4)]
    C = [np.zeros((M, 3), complex) for _ in range(4)]
    RR = [np.zeros((M, M, 3, 3), float) for _ in range(4)]
    CC = [np.zeros((M, M, 3, 3), complex) for _ in range(5)]
    C33 = [np.zeros((M, 3, 3), complex) for _ in range(5)]
    LL = CC[0]
    G = np.zeros((M,), complex)

    SD = SDEProcess(F0)
    SD.Matrix(F0)
    # SD.ItoProcess(10,100,10)


def testFF():
    print(ff(3, 10))


def testRI():
    print(RI(np.array([1 + 2j, -1 + 3j, -5 + 6j])))


def test_gamma_formula():
    u = np.array(
        [[1 + 1j, 2 + 2j, 3 + 3j], [1 - 1j, 2 - 2j, 3 - 3j], [5 - 6j, 7 - 8j, 9 - 10j], [-5 - 6j, -7 - 8j, -9 - 10j]])
    gamma = np.zeros_like(u)
    gamma_formula(u[0], gamma[0])
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
