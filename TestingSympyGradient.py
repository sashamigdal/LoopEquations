import sympy
import numpy as np

x, y = sympy.symbols('x y')
gfg_exp = x + y

exp = sympy.expand(gfg_exp**2)
print("Before Differentiation : {}".format(exp))

# Use sympy.diff() method
dif = sympy.diff(exp, x)

print("After Differentiation : {}".format(dif))


from sympy import MatrixSymbol, Matrix
X = MatrixSymbol('X', 3, 3)
Y = MatrixSymbol('Y', 3, 3)
(X.T*X).I*Y
a = MatrixSymbol("a", 3, 1)
b = MatrixSymbol("b", 3, 1)
test =(a.T*X**2*b).diff(X)

test