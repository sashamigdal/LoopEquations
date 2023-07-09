import quadpy
import numpy as np
def test_Four():
    dim = 4
    scheme = quadpy.sn.dobrodeev_1970(dim)
    val = scheme.integrate(lambda x: np.exp(x[0]), np.zeros(dim), 1.0)
    print(val)