#include <random>
#include <cassert>
#include "Eigen/Core"
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

void TensProd(const Eigen::Vector3Xcd &A, const Eigen::Vector3Xcd &B, Eigen::Matrix3Xcd &AXB)
{
    AXB << A[0] * B[0], A[0] * B[1], A[0] * B[2],
        A[1] * B[0], A[1] * B[1], A[1] * B[2],
        A[2] * B[0], A[2] * B[1], A[2] * B[2];
}

void DetInv(const Eigen::Matrix3Xcd &a,  Eigen::Matrix3Xcd &DIA){
   DIA << -a[2,3]* a[3,2]+a[2,2]* a[3,3],a[1,3]* a[3,2]-a[1,2] *a[3,3],-a[1,3]* a[2,2]+a[1,2] *a[2,3],
   a[2,3] *a[3,1]-a[2,1]* a[3,3],-a[1,3] *a[3,1]+a[1,1]* a[3,3],a[1,3] *a[2,1]-a[1,1]* a[2,3],
   -a[2,2]* a[3,1]+a[2,1] *a[3,2],a[1,2] *a[3,1]-a[1,1] *a[3,2],-a[1,2] *a[2,1]+a[1,1] *a[2,2];
}
complex Det(const Eigen::Matrix3Xcd &a){
    return 
    a[1,3] *(-a[2,2]* a[3,1]+a[2,1]* a[3,2])+
    a[1,2] *(a[2,3] * a[3,1]-a[2,1]* a[3,3])+
    a[1,1] *(-a[2,3]* a[3,2]+a[2,2]* a[3,3]);
}

class MatrixMaker
{
private:
    std::vector<Eigen::Matrix3Xcd> Ak, Bk;
    Eigen::Matrix3Xcd I3;
    std::vector<Eigen::Vector3Xcd> Fk;
    complex ABscale;

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
        size_t M = N_pos + N_neg;
        Ak.resize(M);
        Bk.resize(M);
        Fk.resize(M);
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
            Fk[k] << cs.real, cs.imag, fz;
            walker.Advance();
        }
        Eigen::Matrix3Xcd TensP;
        Eigen::Vector3Xcd DF;
        ABscale = complex(0,0);
        for (; k < M; k++)
        { // k = [0; M)
            //\vec F_k\otimes \vec F_k
            DF = Fk[(k + 1) % M] - Fk[k];
            TensProd(DF, Fk[k], TensP);
            complex FDF = F_k.dot(DF);
            Ak[k] = gamma * I3 + (2 * gamma - 4_i * FDF + 1_i) * TensP;
            Bk[k] = -gamma * I3 + (2 * gamma - 4_i * FDF - 1_i) * TensP;
            TesnProd(DF, DF, TensP);
            //-\gamma +2 i FDF (2 FDF-1)
            Ak[k] += (-gamma + 2_i FDF * (2 * FDF - 1)) * TensP;
            Bk[k] += (+gamma + 2_i FDF * (2 * FDF + 1)) * TensP;
            TesnProd(Fk[k], DF, TensP);
            Ak[k] += 1_i * TensP;
            Bk[k] -= 1_i * TensP;
            ABscale += (Bk[k]-Ak[k]).trace()/6;
        }
    };
    complex Resolvent(complex lambda) const
    {
        // using prepared Ak, Bk, compute R(lambda) and return Tr(R'(lambda)/(R(lambda)-1))
        //    \prod_{i=k}^{i=0} (\hat I*(2\lambda) -\hat A_k )^{-1}(\hat I*(2\lambda) -\hat B_k )
        Eigen::Matrix3Xcd Lam, R1, Tmp2, R, RP;
        Lam = I3 * (2 * lambda);
        R = I3;
        RP.setZero;
        size_t k = 0;
        for (; k < M; k++)
        { // k = [0; M)
            Tmp = (Lam - Ak[k]).inverse();
            Tmp2 = Tmp.dot(Lam - Bk[k]);
            R1 = Tmp2.dot(R);
            RP = 2 * Tmp.dot(R - R1) + Tmp2.dot(RP);
            R = R1;
        }
        return (R - I3).inverse().dot(RP).trace()/3;
    };
    complex CharacteristricPolynomial(complex lambda) const
    {
        // using prepared Ak, Bk, compute R(lambda) and return Tr(R'(lambda)/(R(lambda)-1))
        //    \prod_{i=k}^{i=0} (\hat I*(2\lambda) -\hat A_k )^{-1}(\hat I*(2\lambda) -\hat B_k )
        Eigen::Matrix3Xcd Lam, R1, Tmp2, R;
        Lam = I3 * (2 * lambda);
        R = I3;
        complex PD(1,0);
        size_t k = 0;
        for (; k < M; k++)
        { // k = [0; M)
            Tmp = Lam - Ak[k];
            DetInv(Tmp,Tmp2);
            R = Tmp2.dot(Lam - Bk[k]).dot(R);
            PD = Det(Tmp) * PD;
        }
        return Det(R - PD * I3);
    };
    complex GetScale() const{
        return ABscale;
    }

};
void FindSpectrum(std::int64_t N_pos, std::int64_t N_neg, double beta, complex gamma, complex * lambdas, bool cold_start){
    MatrixMaker mm(N_pos, N_neg, beta,gamma);
    size_t L = 3*(N_pos + N_neg);
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
            complex f= 1/Resolvent(lambdas[k]);
            lambdas[k] -= f;
            errs[k] = abs(f);
        }
    }
    
}