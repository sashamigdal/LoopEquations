#pragma once

#include <math.h>
#include <complex.h>
#include "Eigen/Dense"

using Eigen::Matrix3cd;
using Eigen::Vector3cd;
using namespace std::complex_literals;

#if defined(_MSC_VER)
#   define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#   define EXPORT __attribute__((visibility("default")))
#endif


inline std::complex<double> expi(double a)
{
    double sin_a, cos_a;
    sincos(a, &sin_a, &cos_a);
    return {cos_a, sin_a};
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
    EXPORT double DS( std::int64_t n, std::int64_t m, std::int64_t N_pos, std::int64_t N_neg, double beta, /*OUT*/ double* o_o );
    EXPORT size_t FindSpectrumFromResolvent(std::int64_t N_pos, std::int64_t N_neg, std::uint64_t N_lam,double beta, std::complex<double> gamma,
    /*OUT*/std::complex<double> * lambdas, bool cold_star, double tol);
    EXPORT size_t FindSpectrumFromSparsematrix(std::int64_t N_pos, std::int64_t N_neg, std::uint64_t N_lam, double beta, std::complex<double> gamma,
    /*OUT*/std::complex<double>* lambdas, bool cold_star, double tol);
}
