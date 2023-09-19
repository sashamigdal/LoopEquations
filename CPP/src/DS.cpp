#include <random>
#include <cassert>
#include "Eigen/Dense"
#include "DS.h"

inline complex expi(double a)
{
    double sin_a, cos_a;
    sincos(a, &sin_a, &cos_a);
    return {cos_a, sin_a};
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
/*
TODO: replace formulas by these

\vec \omega_m \cdot \vec \omega_n =
 -\frac{1}{2} \cot ^2\left(\frac{\beta }{2}\right) \sigma _m \sigma _n \sin ^2\left(\frac{1}{4} \left(2 \alpha _m+\beta  \left(\sigma _m-\sigma _n\right)-2 \alpha _n\right)\right)

*/
double DS(std::int64_t n, std::int64_t m, std::int64_t N_pos, std::int64_t N_neg, double beta, /*OUT*/ double *o_o)
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

class MatrixMaker
{
    using Eigen::MatrixX3cd, Eigen::VectorX3cd;
    using complex<double> as complex;
private:
    std::vector<Matrix3Xcd> A, B;
    Matrix3Xcd I3;
    std::vector<Vector3Xcd> F;
    complex ABscale;
    size_t M;

public:
    MatrixMaker(std::int64_t N_pos, std::int64_t N_neg, double beta, complex gamma)
    {
        //     '''
        //     && \hat M_k(\lambda) =
        //    \prod_{i=k}^{i=0} (\hat I -\hat A_k/(2\lambda) )^{-1}(\hat I -\hat B_k /(2\lambda))

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
        double csb = 1 / (2 * eb2.imag);
        complex fz = (0, eb2.real * csb);
        I3.setIdentity();

        std::int64_t k = 0;
        for (; k < M; k++)
        { // k = [0; M)
            cs = expi(walker.get_alpha() * beta) * csb;
            F[k] << cs.real, cs.imag, fz;
            walker.Advance();
        }
        Matrix3Xcd TensP;
        Vector3Xcd DF;
        ABscale = complex(0,0);
        for (; k < M; k++)
        { // k = [0; M)
            //\vec F_k\otimes \vec F_k
            DF = F[(k + 1) % M] - F[k];
            TensP = DF * F[k].transpose();
            complex FDF = F[k].dot(DF);
            A[k] = gamma * I3 + (2. * gamma - 4_i * FDF + 1_i) * TensP;
            B[k] = -gamma * I3 + (2. * gamma - 4_i * FDF - 1_i) * TensP;
            TensP = DF * DF.transpose();
            //-\gamma +2 i FDF (2 FDF-1)
            A[k] += (-gamma + 2_i FDF * (2. * FDF - 1)) * TensP;
            B[k] += (+gamma + 2_i FDF * (2. * FDF + 1)) * TensP;
            TensP = F[k] * DF.transpose();
            A[k] += 1_i * TensP;
            B[k] -= 1_i * TensP;
            ABscale += (B[k]-A[k]).trace()/6.;
        }
    };
    complex Resolvent(complex lambda) const
    {
        // using prepared A, B, compute R(lambda) and return Tr(R'(lambda)/(R(lambda)-1))
        //    \prod_{i=k}^{i=0} (\hat I*(2\lambda) -\hat A_k )^{-1}(\hat I*(2\lambda) -\hat B_k )
        Matrix3Xcd Lam, R, RP, I3,Tmp,Tmp2, R1;
        Lam.setIdentity();
        Lam *= (2. * lambda);
        R.setIdentity();
        I3.setIdentity();
        RP.setZero();
        for (std::int64_t k = 0; k < M; k++)
        { // k = [0; M)
            Tmp = (Lam - A[k]).inverse();
            Tmp2 = Tmp * (Lam - B[k]);
            R1 = Tmp2 * R;
            RP = 2. * Tmp * (R - R1) + Tmp2 * RP;
            R = R1;
        }
        return ((R - I3).inverse()* RP).trace()/3;
    };
    
    complex GetScale() const{
        return ABscale;
    }

    std::int64_t GetSize(){
        return M;
    }

};
void FindSpectrum(std::int64_t N_pos, std::int64_t N_neg, double beta, complex gamma, complex * lambdas, bool cold_start){
    MatrixMaker mm(N_pos, N_neg, beta,gamma);
    size_t M = N_pos + N_neg;
    size_t L = 3*M;
    size_t maxiter=10;
    if(cold_start){
        complex lambda0 = mm.GetScale();
        double r = 10*abs(lambda0);
        for(size_t k=0; k < L; k++){
            lambdas[k] = lambda0 + r * expi(2 * M_PI * k/L);
        }
    }
    
    std::vector<double> errs(L);
    #pragma omp parallel for
    for(size_t k =0; k < L; k++){
        for(size_t iter =0; iter < maxiter; iter++){
            complex f= 1./mm.Resolvent(lambdas[k]);
            lambdas[k] -= f;
            errs[k] = abs(f);
        }
    }
    
}