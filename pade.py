import mpmath as mpm
mpm.dps = 40; mpm.pretty = True
one = mpm.mpf(1)
def f(x):
    return mpm.sqrt((one + 2*x)/(one + x))

def test_Pade():
    a = mpm.taylor(f, 0, 6)
    p, q = mpm.pade(a, 3, 3)
    x = 10
    mpm.polyval(p[::-1], x)/mpm.polyval(q[::-1], x)
    #1.38169105566806
    f(x)
    #1.38169855941551

    A = mpm.taylor(f, 0, 13)
    p,q = mpm.pade(A, 6, 6)

    x = 15
    test =mpm.polyval(p[::-1], x) / mpm.polyval(q[::-1], x)/f(x)

    a = [1/mpm.factorial(n) for n in range(22)]

    p,q = mpm.pade(a,10,11)
    x = -10
    test = mpm.polyval(p[::-1], x) / mpm.polyval(q[::-1], x) * mpm.exp(-x)

    pass

