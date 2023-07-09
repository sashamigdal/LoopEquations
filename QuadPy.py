import quadpy
import numpy as np
from scipy.linalg import eigh
from wolframclient.evaluation import WolframLanguageSession

from Timer import MTimer as Timer
from ScipyQuad import SphericalRestrictedIntegral as SciPySRI
from testMathematica import toMathematica


def test_Four():
    dim = 4
    # scheme = quadpy.un.dobrodeev_1978(dim)
    scheme = quadpy.un.mysovskikh_2(dim)
    def f(x):
        assert (np.min(np.sum(x ** 2, axis=0)) > 0.99)
        return np.heaviside(x[0], 0.5)
    val = scheme.integrate(f, np.zeros(dim), 1.0)/(2*np.pi**2)
    print(val)


def SphericalRestrictedIntegral(R):
    tt, W = eigh(R.imag)
    rr = np.array(R.real, dtype=float)
    RR = W.T @ rr @ W
    dim = 4
    scheme = quadpy.un.mysovskikh_2(dim)
    print("tol=", scheme.test_tolerance)

    vol = scheme.integrate(lambda x: np.ones(x.shape[1]), np.zeros(dim), 1.0)

    def funcC(x):
        XX = x[np.newaxis, :, :] * x[:, np.newaxis, :]
        imtrace = tt.dot(x ** 2)
        # assert(np.min(np.sum(x**2,axis=0)) >0.99)
        return np.exp(1j * np.trace(RR.dot(XX)) - imtrace) / vol * np.heaviside(imtrace, 0.5)

    return scheme.integrate(funcC, np.zeros(dim), 1.0)


def test_GroupIntegral():
    R = np.array([np.random.normal() + 1j * np.random.normal() for _ in range(16)]).reshape((4, 4))
    R += R.T
    with Timer("scipy O(3) Restricted Integral"):
        [rres, ires] = SciPySRI(R)
        print("\nscipy SphericalRestrictedIntegral =", rres[0] + 1j * ires[0] , "+/-" , rres[1] + 1j* ires[1])
    session = WolframLanguageSession()
    session.evaluate('Get["Notebooks/SphericalThetaIntegral.m"]')
    with Timer("Mathematica O(3) Restricted Integral"):
        R_string = toMathematica(R)
        print("\n" + R_string)
        ans = session.evaluate('W[' + R_string + ']')
        print("\nMathematica SphericalThetaIntegral =", ans)

    with Timer("quadpy O(3) Restricted Integral"):
        res = SphericalRestrictedIntegral(R)
        print("\nquadpy: SphericalRestrictedIntegral =", res)
    pass
