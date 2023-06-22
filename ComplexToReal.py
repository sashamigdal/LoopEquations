import numpy as np
from parallel import print_debug

def RealToComplexVec(a, b):
    N = len(b)
    assert len(a) == 2 * N
    a = a.reshape((2, N))
    b[:] = a[0] + 1j * a[1]


def ComplexToRealVec(b, a):
    N = len(b)
    assert len(a) == 2 * N
    a = a.reshape((2, N))
    a[0, :] = b.real
    a[1, :] = b.imag


def RealToComplexMat(A, B):
    N, M = B.shape
    assert A.shape == (2 * N, 2 * M)
    a = A.reshape((2, N, 2, M)).transpose((0, 2, 1, 3))
    B[:] = a[0, 0] + 1j * a[1, 0]

# A = [[Re[b], -Im[b]],[Im[b], Re[b]]
# (b.real + i b.imag).dot(x + i y) = b.real x - b.imag y + i( b.real*y + b.imag* x)
# A.dot([x,y]) = [b.real x - b.imag y ,b.real*y + b.imag* x]

def ComplextoRealMat(B, A):
    N, M = B.shape
    assert A.shape == (2 * N, 2 * M)
    a = A.reshape((2, N, 2, M)).transpose((0, 2, 1, 3))
    a[0, 0] = B.real
    a[0, 1] = -B.imag
    a[1, 0] = B.imag
    a[1, 1] = B.real


def ReformatRealMatrix(A, K):
    assert A.ndim == 2
    N, M = A.shape
    if M == K: return A
    R = np.zeros((N, K), dtype=float)
    if M < K:  # pad with zeros
        R[:, :M] = A
    else:  # M >K, truncate
        R[:, :] = A[:, :K]

    return R
#test

def test_ComplexToRealVec():
    A = np.arange(24, dtype=float)
    B = np.arange(12, dtype=complex) * (1 + 1j)
    R = np.zeros_like(A)
    C = np.zeros_like(B)
    RealToComplexVec(A, C)
    ComplexToRealVec(B, R)
    AA = np.arange(24, dtype=float).reshape((6, 4))
    BB = np.arange(6, dtype=complex).reshape((3, 2))
    RR = np.zeros_like(AA)
    CC = np.zeros_like(BB)
    ComplextoRealMat(BB, RR)
    RealToComplexMat(RR, CC)
    assert (CC == BB).all()
    R = np.arange(4, dtype=float)
    X = RR.dot(R)
    C = np.zeros(2, dtype=complex)
    RealToComplexVec(R, C)
    Y = np.zeros_like(X)
    ComplexToRealVec(CC.dot(C), Y)
    assert (X == Y).all()
    XX = ReformatRealMatrix(RR, 5)
    YY = ReformatRealMatrix(RR, 3)
    pass


def MaxAbsComplexArray(errs):
    mm = [errs.min().real, errs.min().imag, errs.max().real, errs.max().imag]
    return np.max(np.abs(mm))

def SumSqrAbsComplexArray(errs):
    EE = np.conjugate(errs).T.dot(errs).real
    return np.trace(EE) if EE.ndim ==2 else EE