import numpy as np
from wolframclient.evaluation import WolframLanguageSession
from sympy.parsing.mathematica import mathematica
from sympy import var

def test_GroupIntegral():
    R = [-0.1 + 2 * 1.j, -0.2 - 0.01 * 1.j, -0.3 + 0.05 * 1.j, -0.1 - 0.07 * 1.j]
    R_string = ','.join([f"({r.real}) + ({r.imag}) I" for r in R])
    session = WolframLanguageSession()
    session.evaluate('Get["Notebooks/ThetaIntegral.m"]')
    x =session.evaluate('W[{' + R_string + '}]')
    session.terminate()
def test_Lineq():
    session = WolframLanguageSession()
    session.evaluate('Get["Notebooks/MovingFourVertices.m"]')
    Lineq = session.evaluate('Lineq[0]')
    F, dF = var('F dF')
    x =mathematica(repr(Lineq))
    pass


def toMathematicaRaw(val):# a numpy array
    if isinstance(val, np.ndarray) or isinstance(val, list):
        return '{' + ','.join([toMathematicaRaw(x) for x in val]) + '}'
    elif isinstance(val, complex):
        return f"({val.real}) + ({val.imag}) I"
    else:
        return f"({val})"


def toMathematica(array):
    return toMathematicaRaw(array).replace('e', "*^")

# Example 2D array
def test_mathematicaArray():
    array = np.array([[1, 2, 3j], [1e-11 + 4j, 5, 6], [7, 8j, 0]])
    # Convert array to Mathematica string
    mathematica_string = toMathematica(array)
    print(mathematica_string)
