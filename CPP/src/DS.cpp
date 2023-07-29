#include "DS.h"

double DS( std::int64_t n, std::int64_t m, std::int64_t M, std::int64_t * sigmas, double beta ) {
    comp smn, snm;
    int lim = 10000;
    #pragma omp parallel for reduction (+:snm) if( m-n > lim)
        for(int i =n; i <m; i++){
            snm +=  F(sigmas[i], beta);
        }
    #pragma omp parallel for reduction (+:smn) if( n > lim)
        for(int i =0; i <n; i++){
            smn +=  F(sigmas[i], beta);
        }
    #pragma omp parallel for reduction (+:smn) if(M -m > lim)
        for(int i =m; i <M; i++){
            smn +=  F(sigmas[i], beta);
        }
    snm /= double(m-n);
    smn /= double(n + M - m);
	return abs((snm- smn) / (2 * sin(beta / 2)));
}
