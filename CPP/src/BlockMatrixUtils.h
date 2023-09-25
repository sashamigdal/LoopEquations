#include <numeric>
#include "Eigen/Dense" // Eigen::Matrix
#include "arlnsmat.h" // ARluNonSymMatrix

template <typename ARTYPE, typename ARFLOAT>
class ARluNonSymMatrixUnsafe : public ARluNonSymMatrix<ARTYPE,ARFLOAT> {
    template <typename ARTYPE, typename ARFLOAT>
    friend ARluNonSymMatrixUnsafe<ARTYPE,ARFLOAT> ConvertBlockMatrix( const ARluNonSymMatrix<Eigen::Matrix<ARTYPE,3,3>,ARFLOAT>& blockMatrix ) {
        //int nnz = std::accumulate( blockMatrix.a, blockMatrix.a + blockMatrix.nzeros(), 0 );
        int nnz = 9 * blockMatrix.nzeros(); // all elements inside each nonzero 3x3 matrix
        std::vector<Matrix3cd> nzval(nnz);
        std::vector<int> irow(nnz);
        std::vector<int> pcol( 3 * blockMatrix.nrows() + 1 );
        pcol[0] = 0;
        int j = 0;
        for ( size_t col = 0; col != M; col++ ) {
            if ( col != 0 ) {
                irow[j++] = col - 1;
            }

            irow[j++] = col;

            if ( col == 0 ) {
                irow[j++] = M - 1;
            }

            pcol[col + 1] = j;
        }
    }
};
