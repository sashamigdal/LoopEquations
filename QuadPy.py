# import quadpy
import numpy as np
from scipy.linalg import eigh
# from wolframclient.evaluation import WolframLanguageSession

from Timer import MTimer as Timer
from ScipyQuad import SphericalRestrictedIntegral as SciPySRI
from testMathematica import toMathematica


def test_Four():
    dim = 4
    # scheme = quadpy.un.dobrodeev_1978(dim)
    scheme = quadpy.un.mysovskikh_2(dim)
    # print("\n",scheme.degree, "\n",scheme.points)
    # print(scheme.source)
    s = np.random.normal(size=dim)

    def f(x):
        assert x.shape[0] == 4
        assert (np.min(np.sum(x ** 2, axis=0)) > 0.99)
        return np.heaviside(s.dot(x), 0.5)

    val = scheme.integrate(f, np.zeros(dim), 1.0) / (2 * np.pi ** 2)
    print(val)


def SphericalRestrictedIntegral(R):
    dim = 4
    scheme = quadpy.un.mysovskikh_2(dim)

    vol = scheme.integrate(lambda x: np.ones(x.shape[1]), np.zeros(dim), 1.0)

    def funcC(x):
        trc = np.sum(x * (R.dot(x)), axis=0)
        return np.exp(1j * trc) / vol * np.heaviside(trc.imag, 0.5)

    return scheme.integrate(funcC, np.zeros(dim), 1.0)


def GroupIntegral():
    R = np.array([np.random.normal() + 1j * np.random.normal() for _ in range(16)]).reshape((4, 4))
    R += R.T
    with Timer("quadpy O(3) Restricted Integral"):
        res = SphericalRestrictedIntegral(R)
        print("\nquadpy: SphericalRestrictedIntegral =", res)

    with Timer("scipy O(3) Restricted Integral"):
        [rres, ires] = SciPySRI(R)
        print("\nscipy SphericalRestrictedIntegral =", rres[0] + 1j * ires[0], "+/-", rres[1] + 1j * ires[1])
    session = WolframLanguageSession()
    session.evaluate('Get["Notebooks/SphericalThetaIntegral.m"]')
    with Timer("Mathematica O(3) Restricted Integral"):
        R_string = toMathematica(R)
        # print("\n" + R_string)
        ans = session.evaluate('Winv[' + R_string + ']')
        print("\nMathematica Spherical Restricted Integral =", ans)
    session.terminate()

def test_GroupIntegral():
    GroupIntegral()

def SphericalFourierIntegral(R):
    dim = 4
    scheme = quadpy.un.mysovskikh_2(dim)
    # print("tol=", scheme.test_tolerance)
    vol = scheme.integrate(lambda x: np.ones(x.shape[1]), np.zeros(dim), 1.0)

    def func(x):
        return np.exp(1j * np.sum(x * (R.dot(x)), axis=0))/ vol
    return scheme.integrate(func, np.zeros(dim), 1.0)
    '''
    runfile('/home/sasha/LoopEquations/QuadPy.py', wdir='/home/sasha/LoopEquations')
    tol= 1e-14
    quadpy: SphericalFourierIntegral = (0.5802040356506966+0.45746786871519785j)
    quadpy O(3) Fourier Integral 68.31 ms
    '''

def test_FourieO3Integral():
    R = np.array([np.random.normal() for _ in range(16)]).reshape((4, 4))
    R += R.T
    with Timer("quadpy O(3) Fourier Integral"):
        res = SphericalFourierIntegral(R)
        print("\nquadpy: SphericalFourierIntegral =", res)

if __name__ == '__main__':
    R = np.array([np.random.normal() for _ in range(16)]).reshape((4, 4))
    R += R.T
    with Timer("quadpy O(3) Fourier Integral"):
        res = SphericalFourierIntegral(R)
        print("\nquadpy: SphericalFourierIntegral =", res)
