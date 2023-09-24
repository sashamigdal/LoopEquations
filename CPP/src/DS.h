#include <math.h>
#include <complex.h>
#include "Eigen/Dense"

using Eigen::Matrix3cd;
using Eigen::Vector3cd;
using complex = std::complex<double>;
using namespace std::complex_literals;

#if defined(_MSC_VER)
#   define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#   define EXPORT __attribute__((visibility("default")))
#endif

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
    EXPORT size_t FindSpectrumFromResolvent(std::int64_t N_pos, std::int64_t N_neg, std::int64_t N_lam,double beta, std::complex<double> gamma, 
    /*OUT*/std::complex<double> * lambdas, bool cold_star, double tol);
    EXPORT size_t FindSpectrumFromSparsematrix(std::int64_t N_pos, std::int64_t N_neg, std::int64_t N_lam,double beta, std::complex<double> gamma, 
    /*OUT*/std::complex<double> * lambdas, bool cold_star, double tol);
}

class RandomWalker
{
public:
    RandomWalker(std::int64_t N_pos, std::int64_t N_neg) : N_pos(N_pos), N_neg(N_neg), alpha(), gen(std::random_device{}()), unif(0, 1) {}

    int Advance()
    {
        int sigma = RandomSign();
        (sigma == 1 ? N_pos : N_neg)--;
        alpha += sigma;
        return sigma;
    }

    std::int64_t get_alpha() const { return alpha; }

private:
    int RandomSign()
    {
        return (unif(gen) * double(N_pos + N_neg) <= N_neg) ? -1 : 1;
    }

    std::int64_t N_pos;
    std::int64_t N_neg;
    std::int64_t alpha; // alpha[i]
    std::minstd_rand gen;
    std::uniform_real_distribution<double> unif;
};

class MatrixMaker
{
private:
    std::vector<Matrix3cd> A, B;
    Matrix3cd I3;
    std::vector<Vector3cd> F;
    complex ABscale;
    size_t M;

public:
    MatrixMaker(std::int64_t N_pos, std::int64_t N_neg, double beta, complex gamma);
    complex Resolvent(complex lambda) const;

    // Matrix-vector product: w = M*v.
    void MultMv( complex* v, complex* w );

    complex GetScale() const{
        return ABscale;
    }

    std::int64_t GetSize() const{
        return M;
    }

};
