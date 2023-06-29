from scipy import integrate
import numpy as np
from numpy import sin, cos, pi, arcsin, sqrt


def test_simple():
    func = lambda z, y, x: x * y * z ** 4
    print(integrate.tplquad(func, 1, 2, lambda x: 2, lambda x: 3,
                            lambda x, y: 0, lambda x, y: 1))


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

def ThetaIntegral(R):
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
        return np.exp(1j * sum([q[k] ** 2 * R[k] for k in range(4)])) * jac(x, y, z)/(2* pi**2)

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
        if A > 0 and  B < A:
            return arcsin(sqrt(B / A))
        elif A < 0 and  B >= -A and  B < 0:
            return arcsin(sqrt(B / A))
        elif A == 0:
            return pi if B >= 0 else 0
        return pi

    return integrate.tplquad(func, a, b, gfun, hfun, qfun, rfun)

def test_GroupIntegral():
    R = [-0.1 + 2 * 1.j, -0.2 - 0.01 * 1.j, -0.3 + 0.05 * 1.j, -0.1 - 0.07 * 1.j]
    res = ThetaIntegral(R)
    pass