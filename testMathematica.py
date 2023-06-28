
from wolframclient.evaluation import WolframLanguageSession
from sympy.parsing.mathematica import mathematica
from sympy import var

def test_GroupIntegral():
    R = [-0.1 + 2 * 1.j, -0.2 - 0.01 * 1.j, -0.3 + 0.05 * 1.j, -0.1 - 0.07 * 1.j]
    R_string = ','.join([f"({r.real}) + ({r.imag}) I" for r in R])
    session = WolframLanguageSession()
    session.evaluate('Get["Notebooks/RestrictedO3GroupIntegral.m"]')
    x =session.evaluate('W[{' + R_string + '}]')
    session.terminate()
    session = WolframLanguageSession()
    session.evaluate('Get["Notebooks/RestrictedO3GroupIntegral.m"]')
    x = session.evaluate('W[{' + R_string + '}]')
    session.terminate()
def test_Lineq():
    session = WolframLanguageSession()
    session.evaluate('Get["Notebooks/MovingFourVertices.m"]')
    Lineq = session.evaluate('Lineq[0]')
    F, dF = var('F dF')
    x =mathematica(repr(Lineq))
    pass

