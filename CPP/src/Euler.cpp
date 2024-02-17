#include <complex>
#include <cassert>
#include <iostream>
#include "Euler.h"
#include "RandomWalker.h"
#include "MatrixMaker.h"

using Eigen::Matrix3cd;
using Eigen::Vector3cd;
using complex = std::complex<double>;
using namespace std::complex_literals;


double DS(std::int64_t n, std::int64_t m, std::int64_t N_pos, std::int64_t N_neg, std::int64_t q, double beta, /*OUT*/ double *o_o) {
    assert(n < m);
    std::int64_t M = N_pos + N_neg;
    int sigma_n, sigma_m;
    complex S_nm, S_mn;

    RandomWalker walker(N_pos, N_neg);
    std::int64_t i = 0;
    for (; i != n; i++)
    { // i = [0; n)
        S_mn += expi(walker.get_alpha() * beta);
        walker.Advance();
    }

    S_nm += expi(walker.get_alpha() * beta);
    sigma_n = walker.Advance(); // i = n
    for (i++; i != m; i++)
    { // i = (n, m)
        S_nm += expi(walker.get_alpha() * beta);
        walker.Advance();
    }

    S_mn += expi(walker.get_alpha() * beta);
    sigma_m = walker.Advance(); // i = m
    for (i++; i != M; i++)
    { // i = (m, M)
        S_mn += expi(walker.get_alpha() * beta);
        walker.Advance();
    }

    *o_o = -sigma_n * sigma_m / pow(2 * q * tan(beta / 2), 2);

    S_nm /= double(m - n);
    S_mn /= double(n + M - m);
    return abs((S_nm - S_mn) / (2 * sin(beta / 2)));
}

#if 0
size_t FindSpectrumFromResolvent(std::int64_t N_pos, std::int64_t N_neg, std::uint64_t N_lam, double beta, complex gamma, complex * lambdas, bool cold_start, double tol){
    MatrixMaker mm(N_pos, N_neg, beta,gamma);
    size_t M = N_pos + N_neg;
    size_t L =  100;
    size_t maxiter=30;
    if(cold_start){
        //det(R -1) -> det(a/lambda + b/lambda^2) ~ det(a/lambda) det(1 + a^(-1) b/lambda)~ (lambda + tr(a^-1 b))
        // lambda0 = - tr(b/a)
        // a_k = -A_k - B_k
        // b_k = A_k^2 + A_k B_k = -A_k a_k
        // a = Sum{a_k}
        // b = Sum{b_k} + Sum_{k<l}{a_k a_l}
        // Prod = (I + a0 x + b0 x^2) *
        //      . . . . .
        //      * (I + ak x + bk x^2) *
        //      . . . . .
        //      * (I + al x + bl x^2) *
        //       . . . . .
        //      * (I + am x + bm x^2)
        complex lambda0(0.,0);//mm.GetScale();
        std::cout << "lambda0 = " << lambda0 << std::endl;
        double r = 100.;
        for(size_t k=0; k < L; k++){
            lambdas[k] = lambda0 + r * expi(2 * M_PI * k/L);
        }
    }

    std::vector<complex> known_zeros;
    const std::vector<complex> &known_poles = mm.GetPoles();
    // #pragma omp parallel for
    complex dlam = (0,0);
    complex lambda0 = lambdas[0];   
    double r = 10*tol;
    for(size_t k =0; k < L; k++){
        lambdas[k] = lambda0 + r * expi(2 * M_PI * k/L);
        for(size_t iter =0; iter < maxiter; iter++){
            complex R = mm.Resolvent(lambdas[k]);
            for(auto p: known_poles){
                R += complex(1,0)/(lambdas[k] - p);
            }
            for(auto z: known_zeros){
                R -= complex(1,0)/(lambdas[k] - z);
            }
            dlam= complex(1,0)/R;
            lambdas[k] -= dlam;
            if(abs(dlam) < tol){
                known_zeros.push_back(lambdas[k]);
                lambda0 = lambdas[k];
                std::cout << "N=" << M <<  ", k= " << k << ", iter== " << iter << ", found " << lambdas[k] <<" +/-" << abs(dlam) << std::endl;
                break;
            }else{
                // std::cout << "N=" << M <<  ", k= " << k << ", iter== " << iter << ", testing " << lambdas[k] <<" +/-" << abs(dlam) << std::endl;

            }
        }
    }
    if(known_zeros.size() ==0 ) return 0;
    std::sort(known_zeros.begin(), known_zeros.end(),[](auto&a,auto&b){return a.real() == b.real() ? a.imag() < b.imag() : a.real() < b.real();});
    if(known_zeros.size() > N_lam) known_zeros.resize(N_lam);
    for(auto &z: known_zeros){
        z /=  pow(mm.GetSmallFactor(),2);
    }
    std::copy( std::begin(known_zeros), std::end(known_zeros), lambdas );

    return known_zeros.size();
}
#endif

Matrix3cd PseudoInverse(const Matrix3cd &X){
    return (X.adjoint()*X).inverse()* X.adjoint();
}
