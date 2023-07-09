from scipy import integrate
from scipy.linalg import eigh
import numpy as np
from numpy import sin, cos, pi, arcsin, exp, sqrt, prod as fprod, inf, dot
from Timer import MTimer as Timer


def test_simple():
    func3 = lambda z, y, x: x * y * z ** 4
    print(integrate.tplquad(func3, 1, 2, lambda x: 2, lambda x: 3,
                            lambda x, y: 0, lambda x, y: 1))

    func2 = lambda y, x: x * y ** 4
    print(integrate.dblquad(func2, 1, 2, lambda x: 2, lambda x: 3))


'''
Parameters
    ----------
    func : function
        A Python function or method of at least three variables in the
        order (z, y, x).
    a, b : float
        The limits of integration in x: `a` < `b`
    gfun : function or float
        The lower boundary curve in y which is a function taking a single
        floating point argument (x) and returning a floating point result
        or a float indicating a constant boundary curve.
    hfun : function or float
        The upper boundary curve in y (same requirements as `gfun`).
    qfun : function or float
        The lower boundary surface in z.  It must be a function that takes
        two floats in the order (x, y) and returns a float or a float
        indicating a constant boundary surface.
    rfun : function or float
        The upper boundary surface in z. (Same requirements as `qfun`.)
'''


def TripleThetaIntegral(R):
    I = [r.imag for r in R]

    def Q(x, y, z):
        return [sin(z) * sin(y) * cos(x), sin(z) * sin(y) * sin(x), sin(z) * cos(y), cos(z)]

    def s2(x, y):
        return [sin(y) * cos(x), sin(y) * sin(x), cos(y)]

    def jac(x, y, z):
        return sin(z) ** 2 * sin(y)

    a = 0
    b = 2 * pi

    def func(z, y, x):
        q = Q(x, y, z)
        return np.exp(1j * sum([q[k] ** 2 * R[k] for k in range(4)])) * jac(x, y, z) / (2 * pi ** 2)

    def Boundary(x, y, z):
        q = Q(x, y, z)
        return sum([q[k] ** 2 * I[k] for k in range(4)])

    def gfun(x):
        return 0

    def hfun(x):
        return pi

    def qfun(x, y):
        return 0

    def rfun(x, y):
        q = s2(x, y)
        A = I[3] - sum([I[k] * q[k] ** 2 for k in range(3)])
        B = I[3]
        if A > 0 and B < A:
            return arcsin(sqrt(B / A))
        elif A < 0 and B >= -A and B < 0:
            return arcsin(sqrt(B / A))
        elif A == 0:
            return pi if B >= 0 else 0
        return pi

    return integrate.tplquad(func, a, b, gfun, hfun, qfun, rfun)


def F(R, x, y):
    return 1 / sqrt(fprod([x - r - y * r.imag for r in R]))


def SingleThetaIntegral(R):
    def func(x):
        return F(R, x, 0) / (4 * pi ** 3) * exp(1j * x)

    ans = integrate.quad(func, -inf, inf)
    return ans[0] + 1j * ans[1]
    pass


def DoubleThetaIntegral(R):
    def func(y, x):
        return 1 / (2 * pi ** 4) * exp(1j * x) * (F(R, x, y) - F(R, x, -y)) / (2 * y)

    ans = integrate.dblquad(func, -inf, inf, 0, inf)
    return ans[0] + 1j * ans[1]
    pass


'''
W[MM_] /; Dimensions[MM] == {4, 4} :=
  Block[{Q, RR, TT, tt, W, V, QI, u, v, w, Jac, Vol, G, Gc, pc},
   TT = Im[MM];
   {tt, W} = Eigensystem[TT];
   RR = W . Re[MM] . Transpose[W];
   Q = {Sqrt[w] Cos[u], Sqrt[w] Sin[u], Sqrt[1 - w] Cos[v], 
     Sqrt[1 - w] Sin[v]};
   QI = Evaluate[(Q^2) . tt];
   Jac = 1;
   Vol = Integrate[Jac, {w, 0, 1}] (2 Pi)^2;
   pc = PosCond[CoefficientList[QI, w]][[1]];
   (*Print[pc];*)
   G =  Exp[I Q . RR . Q - QI] Boole[pc[[1]] <= w <= pc[[2]]] Jac/Vol;
   Gc = Compile[{{w, _Real}, {u, _Real}, {v, _Real}}, Evaluate[G]];
   (*Print[Gc];*)
   NIntegrate[Gc[w, u, v], {w, 0, 1}, {u, 0, 2 Pi}, {v, 0, 2 Pi}, 
    Method -> "LocalAdaptive", MaxRecursion -> 10, 
    WorkingPrecision -> 10, AccuracyGoal -> 10]
   ];
   '''
'''

 PosCond[{a_ , b_} ] :=
 Module[{x0, region},
  If[b == 0, Return[If[a > 0, {0, 1}, {0, 0}]]];
  x0 = -a/b;
  If[b > 0, region = {Max[0, x0], 1}, region = {0, Min[1, x0]}];
  region]
'''


def PosCond(a, b):
    '''
    a + b z >0 && 0 < z < 1
    '''
    if b == 0:
        return [0, 1] if a > 0 else [0, 0]
    z0 = -a / b
    z0 = max(0, min(1, z0))
    return [z0, 1] if b > 0 else [0, z0]


def SphericalRestrictedIntegral(R):
    tt, W = eigh(R.imag)
    rr = np.array(R.real, dtype=float)
    RR = W @ rr @ W.T

    def funcC(w, v, u):
        Q = np.array([sqrt(w)*cos(u), sqrt(w)*sin(u), sqrt(1-w)*cos(v), sqrt(1-w)*sin(v)], dtype=float)
        return exp(1j * Q.dot(RR.dot(Q)) - tt.dot(Q**2)) / 4 / pi ** 2

    def funcR(w, v, u):
        f = funcC(w, v, u)
        return f.real

    def funcI(w, v, u):
        f = funcC(w, v, u)
        return f.imag

    def Zreg(u, v):
        """
        a + b z = tt[0] z cos^2 x + tt[1] z sin^2 x + tt[2] (1 -z) cos^2 y + tt[3] (1 -z) sin^2 y
        """
        f0 = tt[2] * cos(v)**2 + tt[3]*sin(v)**2
        f1 = tt[0] * cos(u)**2 + tt[1]*sin(u)**2
        return PosCond(f0, f1 - f0)

    def rfun(u, v):
        reg = Zreg(u, v)
        return reg[1]

    def qfun(u, v):
        reg = Zreg(u, v)
        return reg[0]

    eps = 1e-3
    return [integrate.tplquad(funcR, 0, 2 * pi, 0, 2 * pi, qfun, rfun, epsabs=eps),
            integrate.tplquad(funcI, 0, 2 * pi, 0, 2 * pi, qfun, rfun, epsabs=eps)]


def test_GroupIntegral():
    R = np.array([np.random.normal() + 1j * np.random.normal() for _ in range(16)]).reshape((4, 4))
    R += R.T
    with Timer("scipy.quad Theta Restricted Integral"):
        res = SphericalRestrictedIntegral(R)
        # res = SingleThetaIntegral(r) + DoubleThetaIntegral(r)
        print("\nSphericalRestrictedIntegral =", res)
    pass
