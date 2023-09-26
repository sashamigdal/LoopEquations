#include "MatrixMaker.h"
#include "RandomWalker.h"
#include "Euler.h"
#include "BlockMatrixUtils.h"
#include "arlgcomp.h"

using Eigen::Matrix3cd;

MatrixMaker::MatrixMaker(std::int64_t N_pos, std::int64_t N_neg, double beta, complex gamma)
{
    //     '''
    //     && \hat M_k(\lambda) =
    //    \prod_{i=k}^{i=0} (\hat I +\hat A_k/(\lambda) )^{-1}(\hat I -\hat B_k /(\lambda))

    //      &&\hat A_k  =
    //  \gamma  \hat I +(2 \gamma -4 i \vec F_k \cdot\Delta \vec F_k+i) \Delta \vec F_k\otimes \vec F_k\nonumber\\
//     &&\left(-\gamma +2 i (\vec F_k \cdot\Delta \vec F_k) (2 \vec F_k \cdot\Delta \vec F_k-1)\right)\Delta \vec F_k\otimes \Delta \vec F_k +
    //     i \vec F_k \otimes \Delta \vec F_k;\\
//     && \hat B_k =
    //     -\gamma\hat I  +\left(2 \gamma -4 i \vec F_k \cdot\Delta \vec F_k-i\right) \Delta \vec F_k\otimes \vec F_k+\nonumber\\
//     &&\left(\gamma +2 i (\vec F_k \cdot\Delta \vec F_k)(2  \vec F_k \cdot\Delta \vec F_k + 1)\right)\Delta \vec F_k\otimes \Delta \vec F_k
    //     -i \vec F_k \otimes\Delta \vec F_k

    //     \vec F_k =  \frac{1}{2} \csc \left(\frac{\beta }{2}\right) \
// \left\{\cos (\alpha_k), \sin (\alpha_k) \vec w, i \cos \
// \left(\frac{\beta }{2}\right)\right\};
    //     '''
    M = N_pos + N_neg;
    A->resize(M);
    B->resize(M);
    F.resize(M);
    RandomWalker walker(N_pos, N_neg);

    complex cs;
    complex eb2 = expi(beta / 2);
    double csb = 1 / (2 * eb2.imag());
    complex fz = (0, eb2.real() * csb);
    I3.setIdentity();

    std::int64_t k = 0;
    for (; k < M; k++)
    { // k = [0; M)
        cs = expi(walker.get_alpha() * beta) * csb;
        F[k] << cs.real(), cs.imag(), fz;
        walker.Advance();
    }
    Matrix3cd TensP;
    Vector3cd DF;
    Matrix3cd  sumak,sumbk,sumakl;
    sumak.setZero();
    sumbk.setZero();
    sumakl.setZero();
    for (k=0; k < M; k++)
    { // k = [0; M)
        //\vec F_k\otimes \vec F_k
        DF = F[(k + 1) % M] - F[k];
        TensP = DF * F[k].transpose();
        complex FDF = F[k].dot(DF);
        A[k] = gamma * I3 + (2. * gamma - (4.0i * FDF) + 1i) * TensP;
        B[k] = -gamma * I3 + (2. * gamma - 4.0i * FDF - 1i) * TensP;
        TensP = DF * DF.transpose();
        //-\gamma +2 i FDF (2 FDF-1)
        A[k] += (-gamma + 2.0i * FDF * (2. * FDF - 1.0)) * TensP;
        B[k] += (+gamma + 2.0i * FDF * (2. * FDF + 1.0)) * TensP;
        TensP = F[k] * DF.transpose();
        A[k] += 1.0i * TensP;
        B[k] -= 1.0i * TensP;
        // a_k = -A_k - B_k
        // b_k = A_k^2 + A_k B_k = -A_k a_k
        // a = Sum{a_k}
        // b = Sum{b_k} + Sum_{k<l}{a_k a_l}
        Matrix3cd ak =-A[k] - B[k];
        sumbk += (sumak -A[k])* ak;
        sumak += ak;
        // auto sumak_trace = sumak.trace();
    }
    // std::cout << "sumak.determinant() = " << sumak.determinant() << std::endl;

    ABscale = sumbk.norm()/sumak.norm();
};

//void MatrixMaker::MultMv( complex* v, complex* w )
//{
//    //X_k =(M *G)_k =  +B[k] \cdot G_k - A[k] \cdot G_{k+1}
//    //y_k + y_{k+1} = X_k
//    // y_{k+1} = X_k - y_k
//    //M y = B^(-1) A y = lambda y
//    std::int64_t k = 0, L = 3* M;
//    std::vector<Vector3cd> X(M);
//
//// #pragma omp_parallel for
//    for(; k < M; k++){
//        Vector3cd G;
//        X[k].setZero();
//        std::int64_t i,j;
//        i=3 * k;
//        G << v[i],v[i+1],v[i+2];
//        X[k] = B[k] * G;
//        j = (i+3)%L;
//        G << v[j],v[j+1],v[j+2];
//        X[k] -= A[k]* G;
//    }
//    std::vector<Vector3cd> Y(M);
//    Y[0].setZero();
//    // #pragma omp_parallel for
//    for(k=0; k < M; k++){
//        std::int64_t i,j,l;
//        i = 3 * k;
//        for(l=0; l < 3; l++){
//            w[i+l] = Y[k][l];
//        }
//        if(k +1 <M) Y[k+1] = X[k] - Y[k];
//    }
//};

void MatrixMaker::CompEigProbLHSMatrix() {
    std::vector<Matrix3cd> nzval( 2 * M );
    std::vector<int> irow( 2 * M );
    std::vector<int> pcol( M + 1 );
    pcol[0] = 0;
    int j = 0;
    for ( size_t col = 0; col != M; col++ ) {
        if ( col != 0 ) { // upper-diagonal
            nzval[j] = -A[col];
            irow[j++] = col - 1;
        }

        nzval[j] = B[col];
        irow[j++] = col;

        if ( col == 0 ) {
            nzval[j] = -A[col];
            irow[j++] = M - 1;
        }

        pcol[col + 1] = j;
    }
    A.reset();
    B.reset();
    CSC_Matrix<Matrix3cd> arpack_A_block( M, 2 * M, std::move(nzval), std::move(irow), std::move(pcol) );
    arpack_A = ConvertBlockMatrix(arpack_A_block);
};

/* B X
 * X = [g0x   0   0
 *        0 g0y   0
 *        0   0 g0z
 *
 *      g1x   0   0
 *        0 g1y   0
 *        0   0 g1z
 *        ...
 *      g_{n-1}x   0   0
 *        0 gn-1y   0
 *        0   0 gn-1z]
 * Matrix 3n x 3n
 * [.5 .5 0 ...
 * 0  .5 .5 0 ...
 * ...
 * .5 0 ... .5]
 *
 * .5 --> [.5 0 0
 *         0 .5 0
 *         0 0 .5]
 * [1 0 0  1 0 0  0 0 0  0 0 0  0 0 0
 *  0 1 0  0 1 0  0 0 0  0 0 0  0 0 0
 *  0 0 1  0 0 1  0 0 0  0 0 0  0 0 0
 *
 *  0 0 0  1 0 0  1 0 0  0 0 0  0 0 0
 *  0 0 0  0 1 0  0 1 0  0 0 0  0 0 0
 *  0 0 0  0 0 1  0 0 1  0 0 0  0 0 0
 * ...
 *  1 0 0  0 0 0  0 0 0  0 0 0  1 0 0
 *  0 1 0  0 0 0  0 0 0  0 0 0  0 1 0
 *  0 0 1  0 0 0  0 0 0  0 0 0  0 0 1] * 0.5
 * */
void MatrixMaker::CompEigProbRHSMatrix() {
    std::vector<Matrix3cd> nzval( 2 * M, Matrix3cd::Identity() * 0.5 );
    std::vector<int> irow( 2 * M );
    std::vector<int> pcol( M + 1 );
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
    CSC_Matrix<Matrix3cd> arpack_B_block( M, 2 * M, std::move(nzval), std::move(irow), std::move(pcol) );
    arpack_B = ConvertBlockMatrix(arpack_B_block);
};

complex MatrixMaker::Resolvent(complex lambda) const
{
    // using prepared A, B, compute R(lambda) and return Tr(R'(lambda)/(R(lambda)-1))
    //    \prod_{i=k}^{i=0} (\hat I*(\lambda) +\hat A_k )^{-1}(\hat I*(\lambda) -\hat B_k )
    //1/(X+ eps) = 1/X - 1/X eps 1/X + ...
    Matrix3cd Lam, R, RP, I3,Tmp,Tmp2, R1;
    Lam.setIdentity();
    Lam *= lambda;
    R.setIdentity();
    I3.setIdentity();
    RP.setZero();
    for (std::int64_t k = 0; k < M; k++)
    { // k = [0; M)
        Tmp = (Lam+ A[k]).inverse();
        if( Tmp.norm() > 1e10){
            std::cout << " large factor " << Tmp.norm();
        }
        Tmp2 = Tmp * (Lam- B[k]);
        R1 = Tmp2 * R;
        RP = Tmp * (R - R1) + Tmp2 * RP;
        R = R1;
    }
    return ((R - I3).inverse()* RP).trace();
}

int MatrixMaker::FindEigenvalues( std::uint64_t N_lam ) {
    CompEigProbLHSMatrix();
    CompEigProbRHSMatrix();

    EigVal.resize(N_lam);  // Eigenvalues.

    // Defining the eigenvalue problem.
    ARluCompGenEig<double> problem( N_lam, arpack_A, arpack_B, "SR" );

    // Finding eigenvalues.
    complex* pEigVal = &EigVal[0];
    int nFound = problem.Eigenvalues(pEigVal);
    EigVal.resize(nFound);
    std::sort( std::begin(EigVal), std::end(EigVal), [](auto&a,auto&b){
        return a.real() == b.real() ? a.imag() < b.imag() : a.real() < b.real();
    } );
    return nFound;
};
