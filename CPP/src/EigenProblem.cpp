#include "Euler.h"
#include "MatrixMaker.h"

size_t FindSpectrumFromSparsematrix(std::int64_t N_pos, std::int64_t N_neg, std::uint64_t N_lam, double beta, std::complex<double> gamma,
    /*OUT*/std::complex<double> * lambdas, bool cold_start, double tol)
{
    MatrixMaker mm(N_pos, N_neg, beta,gamma);
    size_t M = N_pos + N_neg;
    size_t L = 3*M;
    size_t maxiter=10;
    if(cold_start)
    {
        //det(R -1) -> det(a/lambda + b/lambda^2) ~ det(a/lambda) det(1 + a^(-1) b/lambda)~ (lambda + tr(a^-1 b))
        // lambda0 = - tr(b/a)
        // a_k = -A_k - B_k
        // b_k = A_k^2 + A_k B_k
        // a = Sum{a_k}
        // b = Sum{b_k} + Sum_{k<l}{a_k a_l}
        // Prod = (I + a0 x + b0 x^2) *
        //      . . . . .
        //      * (I + ak x + bk x^2) *
        //      . . . . .
        //      * (I + al x + bl x^2) *
        //       . . . . .
        //      * (I + am x + bm x^2)
        complex lambda0 = mm.GetScale();
        double r = 10*abs(lambda0);
        for(size_t k=0; k < L; k++){
            lambdas[k] = lambda0 + r * expi(2 * M_PI * k/L);
        }
    }

    size_t num = mm.FindEigenvalues(N_lam);
    std::copy( std::begin( mm.Eigenvalues() ), std::end( mm.Eigenvalues() ), lambdas );
    return num;
}
