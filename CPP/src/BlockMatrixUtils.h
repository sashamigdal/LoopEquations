#include <numeric>
#include "Eigen/Dense" // Eigen::Matrix
#include "arlnsmat.h" // ARluNonSymMatrix

template <typename T>
class CSC_Matrix;

template <typename T>
ARluNonSymMatrix<std::complex<T>,T> ConvertBlockMatrix( const CSC_Matrix<Eigen::Matrix<std::complex<T>,3,3>>& blockMatrix );

template <typename T>
class CSC_Matrix {
    template <typename U>
    friend ARluNonSymMatrix<std::complex<U>,U> ConvertBlockMatrix( const CSC_Matrix<Eigen::Matrix<std::complex<U>,3,3>>& blockMatrix );
private:
    int  nnz;
    std::vector<int> irow;
    std::vector<int> pcol;
    std::vector<T>   a;
public:
    CSC_Matrix( int np, int nnzp, std::vector<T>&& ap, std::vector<int>&& irowp, std::vector<int>&& pcolp )
      : nnz(nnzp),
        a( std::move(ap) ),
        irow( std::move(irowp) ),
        pcol( std::move(pcolp) ) {
    }

    size_t ncols() const {
        return pcol.size() - 1;
    }
};

template <typename T>
ARluNonSymMatrix<std::complex<T>,T> ConvertBlockMatrix( const CSC_Matrix<Eigen::Matrix<std::complex<T>,3,3>>& blockMatrix ) {
    std::vector<std::complex<T>> nzval;
    std::vector<int> irow;
    std::vector<int> pcol( 3 * blockMatrix.ncols() + 1 );
    pcol[0] = 0;
    for ( size_t maj_col = 0; maj_col != blockMatrix.ncols(); maj_col++ ) {
        for ( int min_col = 0; min_col != 3; min_col++ ) {
            for ( int i = blockMatrix.pcol[maj_col]; i != blockMatrix.pcol[maj_col + 1]; i++ ) {
                for ( int min_row = 0; min_row != 3; min_row++ ) {
                    int maj_row = blockMatrix.irow[i];
                    const std::complex<T>& elem = blockMatrix.a[i](min_row, min_col);
                    if ( elem != std::complex<T>{} ) {
                        nzval.push_back(elem);
                        irow.push_back( 3 * maj_row + min_row );
                    }
                }
            }
            pcol[3 * maj_col + min_col + 1] = nzval.size();
        }
    }
}
