import functools

import numpy as np
from numpy import cos, sin, pi

import multiprocessing as mp

from ComplexToReal import MaxAbsComplexArray
from parallel import parallel_map, print_debug
from scipy.linalg import null_space

mdot = np.linalg.multi_dot

def circle(k, M):
    delta = pi / M
    return np.array([cos(2 * k * delta), sin(2 * k * delta), 0], dtype=float)


def randomLoop(N, L):
    coeffs = np.random.normal(size=L * 3) + 1j * np.random.normal(size=L * 3)
    FF = coeffs.reshape((3, L))
    a2 = np.kron(np.arange(L), np.arange(N)).reshape((L, N))
    exps = np.exp(a2 * (2.j * pi / N))
    return FF.dot(exps).real.T


def test_RandomLoop():
    C = randomLoop(10, 3)
    pass


def test_multi_dot():
    a = np.arange(6)
    b = np.arange(30).reshape((6, 5))
    c = np.arange(5)
    y = a.dot(b.dot(c))
    pass
    x = mdot([a, b, c])
    assert (x == y).all
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
        C[i] = np.kron(A[i, :], B[i, :]).reshape((d, d))

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
    a = np.arange(12).reshape((4, 3))
    b = np.arange(10, 22).reshape((4, 3))
    c = np.zeros((4, 3, 3))
    parKron22(a, b, c)
    d = VecDot(a, b)
    assert d.ndim == 1
    assert d.shape[0] == 4
    X1 = np.arange(36).reshape((3, 3, 2, 2))
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
    A = np.array(lst).reshape((2, 6))
    B = np.array([E3.dot(q0), E3.dot(q1)]).transpose((1, 0, 2)).reshape((3, 6))
    X = np.vstack([A, B])  # 5 X6
    # X.dot(np.array([dt0,dt1]).reshape(6))  =0
    NS = null_space(X)
    test0 = X.dot(NS)
    Q1 = np.array([E3.dot(q0), Z2]).transpose((1, 0, 2)).reshape((3, 6))
    test1 = Q1.dot(NS)
    Q2 = np.array([Z2, E3.dot(q1)]).transpose((1, 0, 2)).reshape((3, 6))
    test = Q2.dot(NS)
    return NS.T.reshape((-1, 2, 3))


def NullSpace4(F0, F1, F2, F3):
    q0 = F1 - F0
    q1 = F2 - F1
    q2 = F3 - F2
    Ort = np.eye(3)
    # the code in res(l) is imported from Mathematica notebook
    # https: // www.wolframcloud.com / obj / sasha.migdal / Published / FEquationasAnalytic.nb
    # using the small tool math2py.py in this project (thanks to Arthur Migdal)
    def res(l):
        return np.array([np.array([((-4*mdot([F0,E3[l],q0]))+(4*((((1-1j))+(2*mdot([F0,q0]))))*mdot([F0,E3[l],q0]))),((-4*mdot([F0,E3[l],q0])*(-4)*mdot([F3,E3[l],q0]))+(2*((((-1j)*mdot([(-F0),F0])*(-2)*mdot([F0,q0]))+(mdot([F3,F3])*(-2)*mdot([F3,q2]))))*(((-2*mdot([F0,E3[l],q0]))+(2*mdot([F3,E3[l],q0])*(-2)*mdot([q2,E3[l],q0])))))+(4*mdot([q2,E3[l],q0]))),((-4*mdot([F3,E3[l],q0]))+(2*mdot([q2,E3[l],q0]))+(2*((((-1-1j))+(2*mdot([F3,q2]))))*(((-2*mdot([F3,E3[l],q0]))+(2*mdot([q2,E3[l],q0])))))),mdot([Ort[0],E3[l],q0]),mdot([Ort[1],E3[l],q0]),mdot([Ort[2],E3[l],q0])]),np.array([0,((-2*mdot([F0,E3[l],q1]))-(2*mdot([F3,E3[l],q1])*(-2)*mdot([q0,E3[l],q1]))+(2*(((-1j)-(mdot([F0,F0])*(-2)*mdot([F0,q0]))+(mdot([F3,F3])*(-2)*mdot([F3,q2]))))*(((2*mdot([F3,E3[l],q1]))-(2*mdot([q2,E3[l],q1])))))+(2*mdot([q2,E3[l],q1]))),((-4*mdot([F3,E3[l],q1]))+(2*mdot([q2,E3[l],q1]))+(2*((((-1-1j))+(2*mdot([F3,q2]))))*(((-2*mdot([F3,E3[l],q1]))+(2*mdot([q2,E3[l],q1])))))),mdot([Ort[0],E3[l],q1]),mdot([Ort[1],E3[l],q1]),mdot([Ort[2],E3[l],q1])]),np.array([0,0,0,mdot([Ort[0],E3[l],q2]),mdot([Ort[1],E3[l],q2]),mdot([Ort[2],E3[l],q2])])])
    pass
    # (3 projections of ti,3 vectors t0,t1,t2, 6 equations)
    # Eq_i = Mat.transpose().dot(T), T ={\vec t0, \vec t1, \vec t2} = (3,3), Mat = {res(0),res(1), res(2)} (6,3,3)
    X = np.array([res(0), res(1), res(2)], dtype=complex).transpose((2, 1, 0)).reshape((6, 9))
    # test = np.array([q0,q1,q2]).reshape(9)
    NS = null_space(X)
    assert MaxAbsComplexArray(X.dot(NS)) < 1e-10
    NS = NS.reshape((3, 3, -1))
    qd0 = E3.dot(q0)
    qd2 = E3.dot(q2)
    dF1 = mdot([qd0, NS[0]])
    dF2 = -mdot([qd2, NS[2]])
    dF1dF2 = np.vstack([dF1, dF2])
    return dF1dF2


def NullSpace5(F):
    E = np.eye(3).astype(complex)
    Z = np.zeros((3,3),dtype=complex)
    E9 = np.eye(9).astype(complex)
    dF= [None,E9[0:3], E9[3:6], E9[6:9],None]
    X =np.array([(((-2)*mdot([F[0],dF[1]]))+(2*mdot([F[1],dF[1]]))),((-2)*((mdot([F[0],dF[1]])+(mdot([F[1],dF[1]])*((((1+(2*1j)))+(2*mdot([F[0],F[0]]))-(2*mdot([F[1],F[1]])))))))),(2*((mdot([F[1],dF[1]])-mdot([F[1],dF[2]])-mdot([F[2],dF[1]])+mdot([F[2],dF[2]])))),(((-2)*((mdot([F[1],dF[2]])+mdot([F[2],dF[1]])+(mdot([F[2],dF[2]])*((((1+(2*1j)))+(2*mdot([F[1],F[1]]))-(2*mdot([F[2],F[2]]))))))))+(mdot([F[1],dF[1]])*(((((-2)+(4*1j)))+(4*mdot([F[1],F[1]]))-(4*mdot([F[2],F[2]])))))),(2*((mdot([F[2],dF[2]])-mdot([F[2],dF[3]])-mdot([F[3],dF[2]])+mdot([F[3],dF[3]])))),(((-2)*((mdot([F[2],dF[3]])+mdot([F[3],dF[2]])+(mdot([F[3],dF[3]])*((((1+(2*1j)))+(2*mdot([F[2],F[2]]))-(2*mdot([F[3],F[3]]))))))))+(mdot([F[2],dF[2]])*(((((-2)+(4*1j)))+(4*mdot([F[2],F[2]]))-(4*mdot([F[3],F[3]])))))),(2*((mdot([F[3],dF[3]])-mdot([F[4],dF[3]])))),(((-2)*mdot([F[4],dF[3]]))+(mdot([F[3],dF[3]])*(((((-2)+(4*1j)))+(4*mdot([F[3],F[3]]))-(4*mdot([F[4],F[4]]))))))])
    dF1dF2dF3 =null_space(X)
    assert(MaxAbsComplexArray(X.dot(dF1dF2dF3)) < 1e-10)
    return dF1dF2dF3

def RI(X):
    return np.array([X.real, X.imag]).transpose()


def testDot3():
    a = np.random.random(4).reshape((2, 2))
    b = np.random.random(4).reshape((2, 2))
    c = np.zeros((2, 2), dtype=a.dtype)
    np.dot(a, b, c)
    np.dot(a, b, a)
    assert (a == c).all()


def testRI():
    print(RI(np.array([1 + 2j, -1 + 3j, -5 + 6j])))
