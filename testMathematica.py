import numpy as np
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

def test_PadeBorel():
    R = [-0.1 + 2 * 1.j, -0.2 - 0.01 * 1.j, -0.3 + 0.05 * 1.j, -0.1 - 0.07 * 1.j]
    R_string = ','.join([f"({r.real}) + ({r.imag}) I" for r in R])
    session = WolframLanguageSession()
    session.evaluate('Get["/Users/am10485/Documents/Wolfram Mathematica/GroupIntegral.m"]')
    x =session.evaluate('W[{' + R_string + '}]')


    # for t in np.linspace(0., 1., 11):
    #     x = session.evaluate(f'pb[{t}]')
    #     print(f'pb({t}) = {x}')
    # session.evaluate('fff[x_] := x ^ 2')
    # x = session.evaluate('fff[4]')

    session.terminate()
    #0.697175 - 1.15573 *1.j
