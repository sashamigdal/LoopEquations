#include <random>
#include <cassert>
#include "DS.h"

inline complex F( double a ) {
    double sin_a, cos_a;
    sincos( a, &sin_a, &cos_a );
    return {cos_a, sin_a};
}

double Omega( std::int64_t alpha, double beta ) {
    return sin( alpha * beta ) / (2 * (1 - cos(beta)));
}

class RandomWalker {
public:
    RandomWalker( std::int64_t N_pos, std::int64_t N_neg ) : N_pos(N_pos), N_neg(N_neg), alpha(), gen( std::random_device{}() ), unif(0, 1) {}

    void Advance() {
        int sigma = RandomSign();
        (sigma == 1 ? N_pos : N_neg)--;
        alpha += sigma;
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

double DS( std::int64_t n, std::int64_t m, std::int64_t N_pos, std::int64_t N_neg, double beta, /*OUT*/ double* o_o ) {
    assert(n < m);
    std::int64_t M = N_pos + N_neg;
    double Omega_n, Omega_m;
    complex S_nm, S_mn;

    RandomWalker walker( N_pos, N_neg );
    std::int64_t i = 0;
    for ( ; i != n; i++ ) { // i = [0; n)
        walker.Advance();
        S_mn += F( walker.get_alpha() * beta );
    }

    walker.Advance(); // i = n
    S_nm += F( walker.get_alpha() * beta );
    Omega_n = Omega( walker.get_alpha(), beta );

    for ( i++; i != m; i++ ) { // i = (n, m)
        walker.Advance();
        S_nm += F( walker.get_alpha() * beta );
    }

    walker.Advance(); // i = m
    S_mn += F( walker.get_alpha() * beta );
    Omega_m = Omega( walker.get_alpha(), beta );

    for ( i++; i != M; i++ ) { // i = (m, M)
        walker.Advance();
        S_mn += F( walker.get_alpha() * beta );
    }

    *o_o = Omega_n * Omega_m;

    S_nm /= double(m-n);
    S_mn /= double(n + M - m);
    return abs((S_nm - S_mn) / (2 * sin(beta / 2)));
}
