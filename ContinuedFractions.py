from mpmath import mp as mpm

def LogExpansion(f):
    # f(x) =  sum_0^M  f_k x^k + \dots
    # f_0 =1
    # g(x) = log(f(x)) = sum_1^{M} g_k x^k + \dots
    # f(x) g'(x) = f'(x)
    # \sum_{l=0}^{k} f_l (k-l) g_{k-l} = k f_k
    # k g_{k} = k f_{k} -\sum_{l=1}^{k} f_l (k-l) g_{k-l}
    M = len(f)
    g = [mpm.mpc(0) for i in range(M)]
    g[1] = f[1]
    for k in range(2, M):
        lst = [(k - l) * g[k - l] * f[l] for l in range(1, k)]
        g[k] = f[k] - mpm.fsum(lst) / k
    return g


def test_LogExpansion():
    f0 = [1. / mpm.factorial(n) for n in range(10)]
    g = LogExpansion(f0)
    assert (g[1] == mpm.mpc(1))
    lst = [x == mpm.mpc(0) for x in g[2:]]
    print(lst)

def ContinuedFractionCeffs(f):
    depth = len(f)
    a = f[0]
    if depth ==1:
        return [a]
    g = [1/f[1]]
    for l in range(2,depth):
        lst = [g[k]* f[l-k] for k in range(0, l-1)]
        test = mpm.fsum(lst)
        g.append(-test/f[1])
        pass
    ans =  ContinuedFractionCeffs(g)
    ans.append(a)
    return ans


def ValueOfContinuedFraction(coeffs, x):
    L = len(coeffs)
    ans = coeffs[0]
    for k in range(1,L):
        ans = coeffs[k] + x/ans
    pass
    return ans

def test_ContinuedFraction():
    import fractions  # Available in Py2.6 and Py3.0
    def approx2(c, maxd):
        'Fast way using continued fractions'
        return fractions.Fraction.from_float(c).limit_denominator(maxd)
    #1 - x/3 - x^2/45 - (2 x^3)/945 - x^4/4725 - (2 x^5)/93555
    mpm.dps = 30
    f0 = [mpm.mpf(x) for x in [1.,-1./3, -1./45, -2./945, -1./4725, - 2./93555]]
    coeffs = ContinuedFractionCeffs(f0)
    nicefrac = [ approx2(float(x),20) for x in coeffs]
    print(nicefrac)
    # [-11, 9,-7, 5,-3,1]
    value = ValueOfContinuedFraction(coeffs,mpm.mpf(1))
    test = 1/mpm.tan(mpm.mpf(1))
    pass
