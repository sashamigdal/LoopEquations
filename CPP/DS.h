
#define _GNU_SOURCE
#include <math.h>
#include <complex.h>
#define int __int64
#define comp complex<double>
inline comp F( int sigma, double beta){
    beta *= sigma;
    double x,y;
    sincos(beta, &x, &y);
    return comp(x,y)
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

double DS(int n, int m, int M, double * sigmas, double beta) 
{
    comp smn, snm;
    #pragma omp parallel for reduction (+:snm) if( m-n > 10000)
        for(int i =n; i <m; i++){
            snm +=  F(sigmas[i], beta);
        }
    #pragma omp parallel for reduction (+:smn) if( n > 10000)
        for(int i =0; i <n; i++){
            smn +=  F(sigmas[i], beta);
        }
    #pragma omp parallel for reduction (+:snm) if(M -m > 10000)
        for(int i =m; i <M; i++){
            smn +=  F(sigmas[i], beta);
        }
    snm /= double(m-n);
    smn /= double(n + M - m);
    smn -= snm;
    double f = 1 / (2 * sin(beta / 2));
	return smn.Len() * abs(f);
}

