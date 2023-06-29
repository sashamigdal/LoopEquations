from mpmath import quad, quadosc, pi, fprod, exp, expj, sqrt, inf, mpc, mp as mpm, matrix, fdot

from Timer import MTimer as Timer

'''
&&\tilde W(\hat R)= \nonumber \\
&& \int_{-\infty}^{+\infty} \frac{d x\exp{\imath x}}{2 \
    \pi}\left(\prod_{k=1}^4(x  -R_k)\r ight)^{-1/2} H\left(\frac{\Im \
    R_k}{x - R_k}\r ight);\\
&& H(I) = \frac{1}{2} +  \frac{1}{\pi}\int_{0}^{+\infty} d y \frac{\left[\prod_{k=1}^4 \left(1 - y I_k\right)^{-\oh}\right]_-}{y}

integrate.dblquad
Return the double (definite) integral of func(y, x) from x = a..b and y = gfun(x)..hfun(x).

'''


def F(R,x,y):
    return 1 / sqrt(fprod([x - r - y * r.imag for r in R]))

def SingleThetaIntegral(R):
    def func(x):
        return F(R, x, 0) * mpc(1j)/ (4 * pi ) * expj(x)
    ans = quad(func, [-inf, inf], method='tanh-sinh')
    return ans
    pass

def DoubleThetaIntegral(R):
    def func(x,y):
        return 1 / (2 * pi ** 2) * expj(x) * (F(R, x, y) - F(R, x, -y)) / (2 * y)

    ans = quad(func, [-inf, inf],[0, inf],method='tanh-sinh')
    return ans
    pass




def test_GroupIntegral():
    # R = [-0.1 + 2 * 1.j, -0.2 - 0.01 * 1.j, -0.3 + 0.05 * 1.j, -0.1 - 0.07 * 1.j]
    # res = TripleThetaIntegral(R)
    R = [0.00017532-0.00207913j ,-0.00108769+0.00209077j ,-0.00051773+0.00075351j, 0.0014301 -0.00076515j]
    mpm.dps = 20
    r = [mpc(r) for r in R]
    with Timer("Mpmath.quad Theta Integral"):
        res = SingleThetaIntegral(r) + DoubleThetaIntegral(r)
        print("Theta Integral =", res)
    pass