import numpy as np
from numpy import cos, sin, pi
import multiprocessing as mp
from parallel import parallel_map
from scipy.linalg import null_space


def circle(k, M):
    delta = pi / M
    return np.array([cos(2 * k * delta), sin(2 * k * delta), 0], dtype=float)


def randomLoop(N, L):
    coeffs = np.random.normal(size=L * 3) + 1j * np.random.normal(size=L * 3)
    FF = coeffs.reshape(3, L)
    a2 = np.kron(np.arange(L), np.arange(N)).reshape(L, N)
    exps = np.exp(a2 * (2.j * pi / N))
    return FF.dot(exps).real.T


def test_RandomLoop():
    C = randomLoop(10, 3)
    pass


def HodgeDual(v):
    return np.array([
        [0, v[2], -v[1]],
        [-v[2], 0, v[0]],
        [v[1], -v[0], 0]
    ], dtype=v.dtype)


E3 = np.array([HodgeDual(v) for v in np.eye(3)], dtype=float)


def test_LevyCivita():
    a = np.array([1, 2, 3], int)
    b = np.array([4, 5, 6], int)
    ad = a.dot(E3)
    c = ad.dot(b)
    c1 = np.cross(a, b)
    assert (c == c1).all


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

    parallel_map(func, range(N))

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

    parallel_map(func, range(N))


def ZeroBelowEqDiag(X):
    assert X.flags.c_contiguous
    N = len(X)
    assert X.shape[:2] == (N, N)

    def func(k):
        X[k, k:] = 0

    parallel_map(func, range(N))


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


def NullSpace3(F0, F1, F2):
    q0 = F1 - F0
    q1 = F2 - F1
    # dq0 = dt0 X q0 = dt0.dot.E3.dot.q0
    # dq1 = dt1 X q1 = dt1.dot.E3.dot.q1
    # dt0 .dot q0 X F0 = F0 dot dt0 X q0 =0
    # dt1 .dot q1 X F2 = F2 dot dt1 X q1 =0
    # dq0 + dq1 =0
    Z = np.zeros((3), dtype=complex)
    Z2 = np.zeros((3, 3), dtype=complex)
    lst = [[q0.dot(E3.dot(F0)), Z], [Z, q1.dot(E3.dot(F2))]]
    A = np.array(lst).reshape(2, 6)
    B = np.array([E3.dot(q0), E3.dot(q1)]).transpose(1, 0, 2).reshape(3, 6)
    X = np.vstack([A, B])  # 5 X6
    # X.dot(np.array([dt0,dt1]).reshape(6))  =0
    NS = null_space(X)
    test0 = X.dot(NS)
    Q1 = np.array([E3.dot(q0), Z2]).transpose(1, 0, 2).reshape(3, 6)
    test1 = Q1.dot(NS)
    Q2 = np.array([Z2, E3.dot(q1)]).transpose(1, 0, 2).reshape(3, 6)
    test = Q2.dot(NS)
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
    Z2 = np.zeros((3, 3), dtype=complex)
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
    NS = null_space(X)  # 3t* d X K
    # assert MaxAbsComplexArray(X.dot(NS)) < 1e-12
    #
    NS = NS.reshape(3, 3, -1)
    dF1 = qd0.dot(NS[0])  # = dq0
    dF2 = -qd2.dot(NS[2])  # -dq2
    dF1dF2 = np.vstack([dF1, dF2])
    return dF1dF2


def test():
    

    pass


def RI(X):
    return np.array([X.real, X.imag]).transpose()


def testDot3():
    a = np.random.random(4).reshape(2, 2)
    b = np.random.random(4).reshape(2, 2)
    c = np.zeros((2, 2), dtype=a.dtype)
    np.dot(a, b, c)
    np.dot(a, b, a)
    assert (a == c).all()


def testRI():
    print(RI(np.array([1 + 2j, -1 + 3j, -5 + 6j])))
