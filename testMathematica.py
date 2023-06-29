
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

def array_to_string(array):
    rows = len(array)
    cols = len(array[0])

    matrix_string = '{'
    for i in range(rows):
        matrix_string += '{'
        for j in range(cols):
            r = array[i][j]
            matrix_string += f"({r.real}) + ({r.imag}) I".replace("e", "*^")
            if j < cols - 1:
                matrix_string += ','
        matrix_string += '}'
        if i < rows - 1:
            matrix_string += ','
    matrix_string += '}'

    return matrix_string

# Example 2D array
def test_mathematicaArray():
    array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # Convert array to Mathematica string
    mathematica_string = array_to_string(array)
    print(mathematica_string)
