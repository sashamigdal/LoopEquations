#include "DS.h"
#include "arcomp.h"
#include "arrscomp.h"
#include "cmatrixa.h"
#include "rcompsol.h"


template<class T>
void Test(T type)
{

  // Defining a complex matrix.

  CompMatrixA<T> A(10); // n = 10*10.

  // Creating a complex eigenvalue problem and defining what we need:
  // the four eigenvectors of A with largest magnitude.

  ARrcCompStdEig<T> prob(A.ncols(), 4L);

  // Finding an Arnoldi basis.

  while (!prob.ArnoldiBasisFound()) {

    // Calling ARPACK FORTRAN code. Almost all work needed to
    // find an Arnoldi basis is performed by TakeStep.

    prob.TakeStep();

    if ((prob.GetIdo() == 1)||(prob.GetIdo() == -1)) {

      // Performing matrix-vector multiplication.
      // In regular mode, w = Av must be performed whenever
      // GetIdo is equal to 1 or -1. GetVector supplies a pointer
      // to the input vector, v, and PutVector a pointer to the
      // output vector, w.

      A.MultMv(prob.GetVector(), prob.PutVector());

    }

  }

  // Finding eigenvalues and eigenvectors.

  prob.FindEigenvectors();

  // Printing solution.

  Solution(prob);

} // Test.

size_t FindSpectrumFromSparsematrix(std::int64_t N_pos, std::int64_t N_neg, std::uint64_t N_lam,double beta, std::complex<double> gamma,
    /*OUT*/std::complex<double> * lambdas, bool cold_start, double tol)
{
    MatrixMaker mm(N_pos, N_neg, beta,gamma);
    size_t M = N_pos + N_neg;
    size_t L = 3*M;
    size_t maxiter=10;
    if(cold_start)
    {
        //det(R -1) -> det(a/lambda + b/lambda^2) ~ det(a/lambda) det(1 + a^(-1) b/lambda)~ (lambda + tr(a^-1 b))
        // lambda0 = - tr(b/a)
        // a_k = -A_k - B_k
        // b_k = A_k^2 + A_k B_k
        // a = Sum{a_k}
        // b = Sum{b_k} + Sum_{k<l}{a_k a_l}
        // Prod = (I + a0 x + b0 x^2) *
        //      . . . . .
        //      * (I + ak x + bk x^2) *
        //      . . . . .
        //      * (I + al x + bl x^2) *
        //       . . . . .
        //      * (I + am x + bm x^2)
        complex lambda0 = mm.GetScale();
        double r = 10*abs(lambda0);
        for(size_t k=0; k < L; k++){
            lambdas[k] = lambda0 + r * expi(2 * M_PI * k/L);
        }
    }
    
    ARrcCompStdEig<double> prob(L, N_lam,"SR");
    while (!prob.ArnoldiBasisFound()) 
    {

      // Calling ARPACK FORTRAN code. Almost all work needed to
      // find an Arnoldi basis is performed by TakeStep.

      prob.TakeStep();

      if ((prob.GetIdo() == 1)||(prob.GetIdo() == -1)) {

        // Performing matrix-vector multiplication.
        // In regular mode, w = Av must be performed whenever
        // GetIdo is equal to 1 or -1. GetVector supplies a pointer
        // to the input vector, v, and PutVector a pointer to the
        // output vector, w.

        mm.MultMv(prob.GetVector(), prob.PutVector());

      }
    }
    
    // Finding eigenvalues and eigenvectors.

    size_t num =prob.FindEigenvalues();
    if(num ==0 ) return 0;
    complex* known_lambdas = new complex[num];
    prob.Eigenvalues(known_lambdas);
    std::sort( known_lambdas, known_lambdas + num,[](auto&a,auto&b){return a.real() == b.real() ? a.imag() < b.imag() : a.real() < b.real();} );
    num = std::min( num, N_lam );
    std::copy( known_lambdas, known_lambdas + num, lambdas );
    delete[] known_lambdas;
    return num;
}
