
#define _GNU_SOURCE
#include <math.h>
#include <complex.h>

#define DS_API __declspec(dllexport)

using comp = std::complex<double>;

inline comp F( int sigma, double beta){
    beta *= sigma;
    double x,y;
    sincos(beta, &x, &y);
    return comp(x,y);
}

// '''
// def SS(n, m, M, sigmas, beta):
//     Snm = np.sum(F(sigmas[n:m], beta), axis=1)
//     Smn = np.sum(F(sigmas[m:M], beta), axis=1) + np.sum(F(sigmas[0:n], beta), axis=1)
//     snm = Snm / (m - n)
//     smn = Smn / (n + M - m)
//     ds = snm - smn
//     return np.sqrt(ds.dot(ds))
// '''

extern "C"{
    DS_API double DS(std::int64_t n, std::int64_t m, std::int64_t M, double * sigmas, double beta);
}

