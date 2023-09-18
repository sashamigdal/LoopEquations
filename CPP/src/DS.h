#include <math.h>
#include <complex.h>

#if defined(_MSC_VER)
#   define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#   define EXPORT __attribute__((visibility("default")))
#endif

using complex = std::complex<double>;

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
}
'''
&& \vec G_{k+1} =  \hat M_k(\lambda) \cdot \vec G_0;\\
   && \hat M_k(\lambda) = 
   \prod_{i=k}^{i=0} (\hat I -\hat A_k/(2\lambda) )^{-1}(\hat I -\hat B_k /(2\lambda))

 &&\hat A_k  = 
 \gamma  \hat I +(2 \gamma -4 i \vec F_k \cdot\Delta \vec F_k+i) \Delta \vec F_k\otimes \vec F_k\nonumber\\
    &&\left(-\gamma +2 i (\vec F_k \cdot\Delta \vec F_k) (2 \vec F_k \cdot\Delta \vec F_k-1)\right)\Delta \vec F_k\otimes \Delta \vec F_k +
    i \vec F_k \otimes \Delta \vec F_k;\\
    && \hat B_k = 
    -\gamma\hat I  +\left(2 \gamma -4 i \vec F_k \cdot\Delta \vec F_k-i\right) \Delta \vec F_k\otimes \vec F_k+\nonumber\\
    &&\left(\gamma +2 i (\vec F_k \cdot\Delta \vec F_k)(2  \vec F_k \cdot\Delta \vec F_k + 1)\right)\Delta \vec F_k\otimes \Delta \vec F_k 
    -i \vec F_k \otimes\Delta \vec F_k

    \vec F_k =  \frac{1}{2} \csc \left(\frac{\beta }{2}\right) \
\left\{\cos (\alpha_k), \sin (\alpha_k) \vec w, i \cos \
\left(\frac{\beta }{2}\right)\right\};

&&\vec q_k = \sigma_k\{ - \sin \delta_k, \vec w \cos \delta_k , 0\};\\
&&\delta_k =  \alpha_k +\frac{\beta  \sigma_k}{2}
'''
