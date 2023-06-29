from scipy import integrate
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
WW[R_] :=
  Block[{Q, x, y, z},
   V = {x, y, z}; 
   Q = Append[4 V**2, (1 - V . V)**2]/(1 + V . V)**2;
   4/Pi**2 NIntegrate[
     Exp[I Q . R] Boole[Q . Im[R] > 0] 1/(1 + V . V)^3, {x, -Infinity,
       Infinity}, {y, -Infinity, Infinity}, {z, -Infinity, Infinity}]
   ];
   
   
   {{z**2 -> (-2 I3 + I4 - I4 x**2 - I4 y**2 - 
    2 Sqrt[I3**2 - I3 I4 - I1 I4 x**2 + I3 I4 x**2 - I2 I4 y**2 + 
      I3 I4 y**2])/
   I4}, {z**2 -> (-2 I3 + I4 - I4 x**2 - I4 y**2 + 
    2 Sqrt[I3**2 - I3 I4 - I1 I4 x**2 + I3 I4 x**2 - I2 I4 y**2 + 
      I3 I4 y**2])/I4}}
      -((2 I3 + I4 (-1 + x**2 + y**2) + 
  2 Sqrt[I3**2 + I3 I4 (-1 + x**2 + y**2) - I4 (I1 x**2 + I2 y**2)])/I4)
   '''


def StereoIntegral(R):
    [I1, I2, I3, I4] = [r.imag for r in R]
    R = np.array(R, dtype =complex)
    def funcC(z, y, x):
        V = np.array([x, y, z], dtype=float)
        VV = dot(V, V)
        Q = np.array([4 * v * v for v in V] + [(1 - VV) ** 2], dtype=float) / (1 + VV) ** 2
        return exp(1j * dot(Q, R)) * 4 / pi ** 2  / (1 + VV) ** 3

    def rfun(x, y):
        dd = I3 ** 2 + I3 * I4 * (-1 + x ** 2 + y ** 2) - I4 * (I1 * x ** 2 + I2 * y ** 2)
        if dd > 0:
            a = -2 * I3 - I4 * (-1 + x ** 2 + y ** 2)
            Z1 = (a + 2 * sqrt(dd)) / I4
            Z2 = (a - 2 * sqrt(dd)) / I4
            zz = min(Z1, Z2)
            return sqrt(zz) if zz > 0 else inf
        return inf

    def qfun(x, y):
        f = rfun(x, y)
        return -f
    def funcR(z,y,x):
        f = funcC(z,y,x)
        return f.real
    def funcI(z,y,x):
        f = funcC(z,y,x)
        return f.imag

    return integrate.tplquad(funcR, -inf, inf, -inf, inf, qfun, rfun) + 1j * integrate.tplquad(funcI, -inf, inf, -inf, inf, qfun, rfun)


def test_GroupIntegral():
    # R = [-0.1 + 2 * 1.j, -0.2 - 0.01 * 1.j, -0.3 + 0.05 * 1.j, -0.1 - 0.07 * 1.j]
    # res = TripleThetaIntegral(R)
    r = [-0.28174825 - 0.92838393j, 0.15847767 + 0.68931421j, 0.28372187 + 0.77955587j, -0.1604513 - 0.54048615j]

    with Timer("scipy.quad Theta Integral"):
        res = StereoIntegral(r)
        # res = SingleThetaIntegral(r) + DoubleThetaIntegral(r)
        print("StereoIntegral =", res)
    pass
