#pragma once

#include <vector>
#include <memory>
#include "Eigen/Dense"
//#include "arlnsmat.h" // ARluNonSymMatrix

using complex = std::complex<double>;

template <typename T>
class IndexingWrapper : public std::unique_ptr<T> {
public:
    typename T::value_type& operator[] ( size_t index ) {
        return (*std::unique_ptr<T>::get())[index];
    }

    const typename T::value_type& operator[] ( size_t index ) const {
        return (*std::unique_ptr<T>::get())[index];
    }

    IndexingWrapper<T>& operator= ( std::unique_ptr<T>&& other ) noexcept {
        std::unique_ptr<T>::operator= ( std::move(other) );
        return *this;
    }
};

#if 0
class MatrixMaker
{
private:
    IndexingWrapper<std::vector<Eigen::Matrix3cd>> A;
    IndexingWrapper<std::vector<Eigen::Matrix3cd>> B;
    Eigen::Matrix3cd I3;
    std::vector<Eigen::Vector3cd> F;
    complex ABscale;
    size_t M;
    double small_factor;
    //ARluNonSymMatrix<complex,double> arpack_A;
    //ARluNonSymMatrix<complex,double> arpack_B; 
    std::vector<complex> EigVal, Poles;

public:
    MatrixMaker(std::int64_t N_pos, std::int64_t N_neg, double beta, complex gamma);
    complex Resolvent(complex lambda) const;
    int FindEigenvalues( std::uint64_t N_lam );
    const std::vector<complex>& Eigenvalues() const { return EigVal; }
    const std::vector<complex>& GetPoles() const { return Poles; }

    // Matrix-vector product: w = M*v.
    //void MultMv( complex* v, complex* w );

    complex GetScale() const{
        return ABscale;
    }
    double GetSmallFactor() const{
        return small_factor;
    }

    std::int64_t GetSize() const{
        return M;
    }
    
private:
    // A X = lambda B x
    void CompEigProbLHSMatrix();
    void CompEigProbRHSMatrix();
};
#endif
