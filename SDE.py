import numpy as np
from numpy import cos, sin, pi
from scipy.linalg import pinvh, null_space
import sdeint
#no changes, git test
# from parallel import parallel_map
#
#
# # ff[k_, M_] :={1/2 Cos[(2 k \[Pi])/M] Csc[\[Pi]/M],
# #  1/2 Csc[\[Pi]/M] Sin[(2 k \[Pi])/M], 1/2 I Cot[\[Pi]/M]}
#
# def ParallelOp(a, b, Op, K, L, M, S):  # returns (K,M) list of numpy arrays shaped S
#     # a,b are  lists of numpy arrays, Op could be dot or tensordot returning array shaped S
#     def func(k, m):  # returns numpy array shaped S
#         s = Op(a[k, 0], b[0, m])
#         for l in range(1, L, 1):
#             s += Op(a[k, l], b[l, m])
#         return s
#
#     pairs = np.array(np.meshgrid(np.arange(K), np.arange(M))).T.reshape(-1, 2)
#     T = np.concatenate(([K, M], S))
#     return np.array(parallel_map(func, pairs)).reshape(*T)
#

def ff(k, M):
    delta = pi / M
    return np.array([cos(2 * k * delta), sin(2 * k * delta), cos(delta) * 1j]) / (2 * sin(delta))


# RI[X_] := Transpose[ComplexExpand[ReIm[X]]]

def RI(X):
    return np.array([X.real, X.imag]).transpose()


# \[Gamma]formula[u_] := Block[{ V, Q},
# V=ReIm[u];
# Q = PseudoInverse[V.Transpose[V]].V;
# (#[[2]] + I #[[1]])&/@ Q
# ];
# \[Gamma]null[u_] := Block[{ V},
# V=ReIm[u];
# NullSpace[V.Transpose[V]]
# ];

def gamma_formula(u):
    (N, d) = u.shape
    V = np.array([u.real, u.imag]).reshape(2, -1)
    Q = pinvh(V.dot(V.T))
    V1 = Q.dot(V).reshape(2, N, d)
    return V1[1] + V1[0] * 1j


def HodgeDual(v):
    return np.array([
        [0, v[2], -v[1]],
        [-v[2], 0, v[0]],
        [v[1], -v[0], 0]
    ], dtype=v.dtype)


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
#V =2 G[[#]]*q[[#]]-F[[#]] &/@ Range[M];
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

class SDEMatrices():
    def __init__(self, F):
        M = len(F)
        q = np.roll(F, 1, axis=0) - F
        qd = np.array([HodgeDual(v) for v in q])
        G = np.array([2 * F[k].dot(q[k]) - 1j for k in range(M)])
        U = np.array([-G[k]*F[k].dot(qd[k]) for k in range(M)])
        gamma = gamma_formula(U)
        V = np.array([2*G[k]*q[k]-F[k] for k in range(M)])
        gammaXV = np.array([np.kron(gamma[k],V[k]) for k in range(M)]).reshape(M,3,3)
        LL =   gammaXV.reshape(M*3,3).dot(qd.reshape(M*3,3).T).reshape(M,3,M,3).transpose((0,2,1,3))
        X = LL.real
        for k in range(M):
            X[k,k] -= np.kron(gamma[k], U[k]).real.reshape(3,3)
        X = X.transpose((0,2,1,3)).reshape(3*M,3*M)
        MM = X
        LLI = LL.imag.transpose((0,2,1,3)).reshape(3*M,3*M)
        for _ in range(M-1):
            MM = X - LLI.dot(MM)
        pass
        Y = np.array([np.kron(gamma[k], V[k]).reshape(3,3) for k in range(M)]).reshape(3*M,3)
        PP =Y.real
        QQ = -Y.imag
        for _ in range(M-1):
            PP = Y.real + LLI.dot(PP)
            QQ =-Y.imag + LLI.dot(QQ)
        pass
        QD =qd.transpose((1,0,2)).reshape(3,3*M)
        P = QD.dot(PP)
        Q = QD.dot(QQ)
        BB = np.array([[P.real,Q.real],[P.imag,Q.imag]]).transpose(0,2,1,3).reshape(6,6)
        NS = null_space(BB).reshape(2,3,-1)
        NS = (NS[0]-1j*NS[1]).transpose()
        BB = pinvh(BB)
        K = len(NS)
        PQM = BB.reshape(2,3,2,3).transpose((0,2,1,3))
        X = PQM[0,0] +PQM[1,1] - 1j *PQM[1,0] + 1j*PQM[0,1]
        Z = X.dot(qd.transpose(1,2,0).reshape(3,3*M))
        Lambda = -Z.imag - Y.T.dot(MM)
        Z = NS.dot(QD)
        Theta =Z.imag - Z.real.dot(MM)
        CC = Theta.dot(Theta.T)
        # eig=np.linalg.eigh(CC)
        CCI = np.linalg.inv(CC)
        Y = CCI.dot(Theta)
        Lambda -= Lambda.dot(Theta.T).dot(Y)
        # test =Lambda.dot(Theta.T)
        # test
        TT = MM - MM.dot(Theta.T).dot(Y)
        TT = TT.astype(complex).reshape(M,3,M,3).transpose((0,2,1,3))
        TT[0] += Lambda.reshape(3,3,M).transpose((2,0,1))
        for k in range(M-1):
            TT[k+1] += TT[k]
        self.B =TT.transpose((0,2,1,3)).reshape(3*M,3*M)
        self.B

############### tests
def testSDE():
    M=10
    F = np.array([ff(k, M) for k in range(M)])
    XX = SDEMatrices(F)
    XX.B


def testFF():
    print(ff(3, 10))


def testRI():
    print(RI(np.array([1 + 2j, -1 + 3j, -5 + 6j])))


def test_gamma_formula():
    u = np.array(
        [[1 + 1j, 2 + 2j, 3 + 3j], [1 - 1j, 2 - 2j, 3 - 3j], [5 - 6j, 7 - 8j, 9 - 10j], [-5 - 6j, -7 - 8j, -9 - 10j]])
    gamma = gamma_formula(u)
    u

def testtensorDot():
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    b = 10* a
    c =np.tensordot(a,b,axes=([2],[2]))
    a1 = a.reshape(4,  3)
    b1 = b.reshape(4, 3)
    c1 = a1.dot(b1.T).reshape(2,2,2,2)
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
