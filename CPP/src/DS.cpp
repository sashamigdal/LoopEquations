#include <complex>
#include <random>
#include <cassert>
#include <iostream>
#include "DS.h"

using Eigen::Matrix3cd;
using Eigen::Vector3cd;
using complex = std::complex<double>;
using namespace std::complex_literals;

inline complex expi(double a)
{
    double sin_a, cos_a;
    sincos(a, &sin_a, &cos_a);
    return {cos_a, sin_a};
}
/*
TODO: replace formulas by these

\vec \omega_m \cdot \vec \omega_n =
 -\frac{1}{2} \cot ^2\left(\frac{\beta }{2}\right) \sigma _m \sigma _n \sin ^2\left(\frac{1}{4} \left(2 \alpha _m+\beta  \left(\sigma _m-\sigma _n\right)-2 \alpha _n\right)\right)

*/
double DS(std::int64_t n, std::int64_t m, std::int64_t N_pos, std::int64_t N_neg, std::int64_t N_lam, double beta, /*OUT*/ double *o_o)
{
    assert(n < m);
    std::int64_t M = N_pos + N_neg;
    int sigma_n, sigma_m, alpha_m, alpha_n;
    complex S_nm, S_mn;
    double R_nm;

    RandomWalker walker(N_pos, N_neg);
    std::int64_t i = 0;
    for (; i != n; i++)
    { // i = [0; n)
        S_mn += expi(walker.get_alpha() * beta);
        walker.Advance();
    }

    alpha_n = walker.get_alpha();
    S_nm += expi(alpha_n * beta);
    sigma_n = walker.Advance(); // i = n
    for (i++; i != m; i++)
    { // i = (n, m)
        S_nm += expi(walker.get_alpha() * beta);
        walker.Advance();
    }

    alpha_m = walker.get_alpha();
    S_mn += expi(alpha_m * beta);
    sigma_m = walker.Advance(); // i = m
    for (i++; i != M; i++)
    { // i = (m, M)
        S_mn += expi(walker.get_alpha() * beta);
        walker.Advance();
    }

    /*
    -\frac{1}{2} \cot ^2\left(\frac{\beta }{2}\right) \sigma _m \sigma _n \sin ^2\left(\frac{1}{4} \left(2 \alpha _m+\beta  \left(\sigma _m-\sigma _n\right)-2 \alpha _n\right)\right)
    */
    *o_o = -M * (M - 1) / 2 * sigma_n * sigma_m / (2 * pow(tan(beta / 2), 2)) * pow(sin(beta / 4 * (2 * (alpha_m - alpha_n) + sigma_m - sigma_n)), 2);

    S_nm /= double(m - n);
    S_mn /= double(n + M - m);
    return abs((S_nm - S_mn) / (2 * sin(beta / 2)));
}

size_t FindSpectrumFromResolvent(std::int64_t N_pos, std::int64_t N_neg, std::int64_t N_lam, double beta, complex gamma, complex * lambdas, bool cold_start, double tol){
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

    std::vector<complex> known_lambdas;
    // #pragma omp parallel for
    complex dlam = (0,0);
    complex lambda0 = lambdas[0];
    double r = 10*tol;
    for(size_t k =0; k < L; k++){
        lambdas[k] = lambda0 + r * expi(2 * M_PI * k/L);
        for(size_t iter =0; iter < maxiter; iter++){
            complex R = mm.Resolvent(lambdas[k]);
            for(auto lam: known_lambdas){
                R -= complex(1,0)/(lambdas[k] - lam);
            }
            dlam= complex(1,0)/R;
            lambdas[k] -= dlam;
            if(abs(dlam) < tol){
                known_lambdas.push_back(lambdas[k]);
                break;
            }
        }
        // std::cout << "N=" << M <<  ", k= " << k << ", found " << lambdas[k] <<" +/-" << abs(dlam) << std::endl;
    }
    if(known_lambdas.size() ==0 ) return 0;
    std::sort(known_lambdas.begin(), known_lambdas.end(),[](auto&a,auto&b){return a.real() == b.real() ? a.imag() < b.imag() : a.real() < b.real();});
    if(known_lambdas.size() > N_lam) known_lambdas.resize(N_lam);
    std::copy( std::begin(known_lambdas), std::end(known_lambdas), lambdas );
    return known_lambdas.size();
}
Matrix3cd PseudoInverse(const Matrix3cd &X){
    return (X.adjoint()*X).inverse()* X.adjoint();
}

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
    A.resize(M);
    B.resize(M);
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



void MatrixMaker::MultMv( complex* v, complex* w )
{
    //X_k =(M *G)_k =  +B[k] \cdot G_k - A[k] \cdot G_{k+1}
    //y_k + y_{k+1} = X_k
    // y_{k+1} = X_k - y_k
    //M y = B^(-1) A y = lambda y
    std::int64_t k = 0, L = 3* M;
    std::vector<Vector3cd> X(M);
   
// #pragma omp_parallel for
    for(; k < M; k++){
        Vector3cd G;
        X[k].setZero();
        std::int64_t i,j;
        i=3 * k;
        G << v[i],v[i+1],v[i+2];
        X[k] = B[k] * G;
        j = (i+3)%L;
        G << v[j],v[j+1],v[j+2];
        X[k] -= A[k]* G;
    }
    std::vector<Vector3cd> Y(M);
    Y[0].setZero();
    // #pragma omp_parallel for
    for(k=0; k < M; k++){
        std::int64_t i,j,l;
        i = 3 * k;
        for(l=0; l < 3; l++){
            w[i+l] = Y[k][l];
        }
        if(k +1 <M) Y[k+1] = X[k] - Y[k];
    }
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
};

   