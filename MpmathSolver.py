from mpmath import mp as mpm

mpm.dps = 30
prec = 20

f_1 = lambda x, y: 1623.66790917 * x ** 2 + 468.829686367 * x * y + 252.762128419 * y ** 2 + -1027209.42116 * x + -301192.975791 * y + 188804356.212

f_2 = lambda x, y: 11154.1759415 * x ** 2 + 31741.0229155 * x * y + 32933.5622632 * y ** 2 + -16226174.4037 * x + -26323622.7497 * y + 6038609721.67

def solve(f1, f2, initxy):
    return mpm.findroot([f1, f2], initxy, solver='muller')

def show(x, y):
    print('x=', mpm.nstr(x, prec))
    print('y=', mpm.nstr(y, prec))
    print( mpm.nstr(f_1(x, y), prec))
    print(mpm.nstr(f_2(x, y), prec))


f1a = f_1
f2a = f_2
xa, ya = solve(f1a, f2a, (240+40j, 265-85j))
show(xa, ya)

f1b  = lambda x, y: f1a(x, y) / ((x - xa) * (y - ya))
f2b  = lambda x, y: f2a(x, y) / ((x - xa) * (y - ya))
xb, yb = solve(f1b, f2b, (290+20j, 270+30j))
show(xb, yb)

