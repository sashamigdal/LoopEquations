import numpy as np

# Python program to find
# the nth number in Calkin
# Wilf sequence:
frac = [0, 1]
# returns 1x2 int array
# which contains the nth
# rational number
def nthRational(n):
    if n > 0:
        nthRational(int(n / 2))
    # ~n&1 is equivalent to
    # !n%2?1:0 and n&1 is
    frac[~n & 1] += frac[n & 1]
    return frac
# Driver code

def RationalRandom(M):
    a,b = (0,1)
    while True:
        den = a + b
        if den >= M: 
            break
        if np.random.randint(2) ==1:
            b = den
        else:
            a = den
    return (a,b) if a < b else (b,a)

if __name__ == "__main__":
    M = 1111025 # testing for n=13
    # converting array
    # to string format
    print(RationalRandom(M))
# This code is contributed