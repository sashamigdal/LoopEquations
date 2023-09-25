#include <vector>
#include "Eigen/Dense"

using complex = std::complex<double>;

class MatrixMaker
{
private:
    std::vector<Eigen::Matrix3cd> A, B;
    Eigen::Matrix3cd I3;
    std::vector<Eigen::Vector3cd> F;
    complex ABscale;
    size_t M;

public:
    MatrixMaker(std::int64_t N_pos, std::int64_t N_neg, double beta, complex gamma);
    complex Resolvent(complex lambda) const;

    // Matrix-vector product: w = M*v.
    //void MultMv( complex* v, complex* w );

    complex GetScale() const{
        return ABscale;
    }

    std::int64_t GetSize() const{
        return M;
    }
private:
    // A X = lambda B x
    void CompEigProbLHSMatrix();
    void CompEigProbRHSMatrix();
};
