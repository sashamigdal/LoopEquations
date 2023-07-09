import quadpy
import numpy as np
from scipy.linalg import eigh
from Timer import MTimer as Timer


def test_Four():
    dim = 4
    scheme = quadpy.sn.dobrodeev_1970(dim)
    val = scheme.integrate(lambda x: np.ones(73), np.zeros(dim), 1.0)
    print(val)

def SphericalRestrictedIntegral(R):
    tt, W = eigh(R.imag)
    rr = np.array(R.real, dtype=float)
    RR = W.T @ rr @ W
    dim = 4
    scheme = quadpy.sn.dobrodeev_1970(dim)
    vol = scheme.integrate(lambda x: np.ones(73), np.zeros(dim), 1.0)
    def funcC(x):
        XX = x[np.newaxis, :, :] * x[:, np.newaxis, :]
        imtrace = tt.dot(x ** 2)
        return np.exp(1j * np.trace(RR.dot(XX)) - imtrace) * np.heaviside(imtrace, 0.5) / vol

    return scheme.integrate(funcC, np.zeros(dim), 1.0)


def test_GroupIntegral():
    R = np.array([np.random.normal() + 1j * np.random.normal() for _ in range(16)]).reshape((4, 4))
    R += R.T
    with Timer("quadpy O(3) Restricted Integral"):
        res = SphericalRestrictedIntegral(R)
        print("\nSphericalRestrictedIntegral =", res)
    pass
