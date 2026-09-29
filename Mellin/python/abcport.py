"""Python port of ABCInterpolator.wl (Migdal). mpmath-based, principal branches as in Mathematica."""
from mpmath import mp, mpf, mpc, sqrt, cos, sin, tan, log, exp, pi, re, findroot, quad, inf, diff
mp.dps = 30

def gg(r, D):
    a = sqrt(r*(D-1)); b = sqrt(r*D)
    return (-2 + 2*cos(a)*cos(b) + ((2*D-1)*sin(a)*sin(b))/sqrt((D-1)*D))/r**2

def action(r, D):
    sr = sqrt(r); s1 = sqrt(-1+D); s0 = sqrt(D)
    num = sr*(-8*sr*s1*s0*cos(sr*(s1-s0)) + 8*sr*s1*s0*cos(sr*(s1+s0))
        + 4*sr*(-1+2*D - D*cos(2*sr*s1) - (-1+D)*cos(2*sr*s0))
        - 8*s1*D*sin(sr*(s1-s0)) - s1*sin(2*sr*(s1-s0)) + 4*s1*D*sin(2*sr*(s1-s0))
        - 8*s1*D*sin(sr*(s1+s0)) - s1*sin(2*sr*(s1+s0)) + 4*s1*D*sin(2*sr*(s1+s0))
        + 2*s1*sin(2*sr*s1)
        + s0*(-8*(-1+D)*sin(sr*(s1-s0)) + (-3+4*D)*sin(2*sr*(s1-s0)) + 8*(-1+D)*sin(sr*(s1+s0))
              + (3-4*D)*sin(2*sr*(s1+s0)) + 2*sin(2*sr*s0)))
    return num/(16*(-1+D)*D*(cos(sr*s1)-cos(sr*s0))**2)

def cons(D, r):
    sr = sqrt(r); s1 = sqrt(-1+D); s0 = sqrt(D)
    inner = (2*D*cos(2*sr*s1) + (1/sr)*(2*sr*(1-2*D) - 4*s1*D*sin(sr*(s1-s0)) - 4*s1*D*sin(sr*(s1+s0))
             + s1*sin(2*sr*s1) + cos(2*sr*s0)*(2*sr*(-1+D) + s1*(-1+4*D)*sin(2*sr*s1))
             + 8*sr*s1*s0*sin(sr*s1)*sin(sr*s0)
             + s0*(8*(-1+D)*cos(sr*s1)*sin(sr*s0) + (1+(3-4*D)*cos(2*sr*s1))*sin(2*sr*s0))))
    return -inner/(16*(-1+D)*D*(cos(sr*s1)-cos(sr*s0))**2)

def alal(D, r):
    a = sqrt(r*(-1+D)); b = sqrt(r*D); c = sqrt((-1+D)*D)
    return (r*(D*sin(a) - c*sin(b))*(D*cos(b)*sin(a) - c*cos(a)*sin(b)))/((-1+D)*D**2*(cos(a)-cos(b))**2)

def fOm(w, r, D):
    sw = sqrt(w)
    return -r*cos(sw/2) + r*cos((1-2*D)*sw/2) + (-1+D)*D*sw*(r+(-1+D)*D*w)*sin(sw/2)

def r2(D, guess=None):
    """Root of gg(r,D) continued from r=44.7466 at D=0.5 (interp2 in the .wl); bracketed so it cannot jump branch."""
    g = mpf('44.74657089612265') if guess is None else mpf(guess)
    h = lambda r: re(gg(r, D))
    for w in (mpf('0.05'), mpf('0.2'), mpf('0.5'), mpf('1'), mpf('2')):
        a, b = g - w, g + w
        if h(a)*h(b) < 0:
            return findroot(h, (a, b), solver='illinois', tol=mpf(10)**(-2*mp.dps//3))
    raise ValueError(f'no bracket for root near {g} at D={D}')

def IQ(D, r=None, eps=mpf('0.1')):
    """IQ[D] of ABCInterpolator.wl with the omega-derivative done analytically (s = Sqrt[omega])."""
    r = r2(D) if r is None else r
    a = 1 - 2*D; k = (D - 1)*D
    f_s = lambda s: -r*cos(s/2) + r*cos(a*s/2) + k*s*(r + k*s**2)*sin(s/2)
    fp_s = lambda s: (r*sin(s/2)/2 - r*a*sin(a*s/2)/2
                      + k*((r + k*s**2)*sin(s/2) + 2*k*s**2*sin(s/2) + s*(r + k*s**2)*cos(s/2)/2))
    def Phi(w):
        s = sqrt(w)
        return fp_s(s)/(2*s)/f_s(s) + tan(s/2)/(4*s) - mpf(3)/(2*w)
    g = lambda x: re(Phi(eps + 1j*x)*log(eps + 1j*x))
    pts = [0] + [mpf(i)/20 for i in range(1, 41)] + [3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 200, 500,
                                                     1000, 10**4, 10**5, 10**6, inf]
    return 2*quad(g, pts)  # integrand is even in x

def ABC(D, r=None, iq=None):
    r = r2(D) if r is None else r
    L = re(action(r, D)); S = re(alal(D, r)); J = re(cons(D, r))
    Q = exp((IQ(D, r) if iq is None else iq)/(8*pi))
    return Q*2*(r-6)/(r+12), Q*J/S, Q*L/(2*pi*S)

D1 = mpf('0.15714261196307505'); D2 = mpf('0.4301495990065511')

