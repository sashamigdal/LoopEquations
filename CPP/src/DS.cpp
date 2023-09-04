#include <random>
#include <cassert>
#include "DS.h"

inline complex expi( double a ) {
    double sin_a, cos_a;
    sincos( a, &sin_a, &cos_a );
    return {cos_a, sin_a};
}


class RandomWalker {
public:
    RandomWalker( std::int64_t N_pos, std::int64_t N_neg ) : N_pos(N_pos), N_neg(N_neg), alpha(), gen( std::random_device{}() ), unif(0, 1) {}

    int Advance() {
        int sigma = RandomSign();
        (sigma == 1 ? N_pos : N_neg)--;
        alpha += sigma;
        return sigma;
    }

    std::int64_t get_alpha() const { return alpha; }
private:
    int RandomSign() {
        return (unif(gen) * double(N_pos + N_neg) <= N_neg) ? -1 : 1;
    }

    std::int64_t N_pos;
    std::int64_t N_neg;
    std::int64_t alpha; // alpha[i]
    std::minstd_rand gen;
    std::uniform_real_distribution<double> unif;
};
/*
TODO: replace formulas by these

\vec \omega_m \cdot \vec \omega_n =
 -\frac{1}{2} \cot ^2\left(\frac{\beta }{2}\right) \sigma _m \sigma _n \sin ^2\left(\frac{1}{4} \left(2 \alpha _m+\beta  \left(\sigma _m-\sigma _n\right)-2 \alpha _n\right)\right)

*/
double DS( std::int64_t n, std::int64_t m, std::int64_t N_pos, std::int64_t N_neg, double beta, /*OUT*/ double* o_o ) {
    assert(n < m);
    std::int64_t M = N_pos + N_neg;
    int sigma_n, sigma_m, alpha_m, alpha_n;
    complex S_nm, S_mn;
    double R_nm;
    
    RandomWalker walker( N_pos, N_neg );
    std::int64_t i = 0;
    for ( ; i != n; i++ ) { // i = [0; n)
        S_mn += expi( walker.get_alpha() * beta );
        walker.Advance();
    }

    
    alpha_n = walker.get_alpha();
    S_nm += expi( alpha_n * beta );
    sigma_n = walker.Advance(); // i = n
    for ( i++; i != m; i++ ) { // i = (n, m)
        S_nm += expi( walker.get_alpha() * beta );
        walker.Advance();
        
    }

    alpha_m = walker.get_alpha();
    S_mn += expi(alpha_m * beta );
    sigma_m = walker.Advance(); // i = m
    for ( i++; i != M; i++ ) { // i = (m, M)
        walker.Advance();
        S_mn += expi( walker.get_alpha() * beta );
    }

/*
-\frac{1}{2} \cot ^2\left(\frac{\beta }{2}\right) \sigma _m \sigma _n \sin ^2\left(\frac{1}{4} \left(2 \alpha _m+\beta  \left(\sigma _m-\sigma _n\right)-2 \alpha _n\right)\right)
*/
    *o_o = - M*(M-1)/2 *sigma_n*sigma_m/(2*pow(tan(beta/2),2))* pow(sin(beta/4* (2 * (alpha_m -alpha_n)+  sigma_m-sigma_n )),2);

    S_nm /= double(m-n);
    S_mn /= double(n + M - m);
    return abs((S_nm - S_mn) / (2 * sin(beta / 2)));
}
