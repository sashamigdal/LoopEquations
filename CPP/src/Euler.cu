#include <cassert>
#include <utility>
#include <random>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h> // for Intellisense
#include <cuComplex.h>

using namespace std::string_literals;

// no noticeable difference
#define USE_DOUBLE 1

#if USE_DOUBLE
using cuAnyComplex = cuDoubleComplex;
using real = double;
#   define make_cuComplex make_cuDoubleComplex
#   define cuCrealr cuCreal
#   define cuCimagr cuCimag
#   define cuCsubr cuCsub
#   define cuCabsr cuCabs
#   define curand_uniform_real curand_uniform_double
#else
using cuAnyComplex = cuFloatComplex;
using real = float;
#   define make_cuComplex make_cuFloatComplex
#   define cuCrealr cuCrealf
#   define cuCimagr cuCimagf
#   define cuCsubr cuCsubf
#   define cuCabsr cuCabsf
#   define curand_uniform_real curand_uniform
#endif

void CheckErrorCode( cudaError_t err_code ) {
    if ( err_code ) {
        std::cerr << "CUDA ERROR " << err_code << " happened" << std::endl;
    }
}

__device__ static inline cuAnyComplex expi( real a ) {
    real sin_a, cos_a;
    sincos(a, &sin_a, &cos_a);
    return make_cuComplex( cos_a, sin_a );
}

__device__ static inline cuAnyComplex& operator+= ( cuAnyComplex& z, cuAnyComplex w ) {
    z.x += w.x;
    z.y += w.y;
    return z;
}

__device__ static inline cuAnyComplex operator- ( cuAnyComplex z, cuAnyComplex w ) {
    return cuCsubr( z, w );
}

__device__ static inline cuAnyComplex operator/ ( cuAnyComplex z, real a ) {
    return make_cuComplex( cuCrealr(z) / a, cuCimagr(z) / a );
}

__device__ static inline cuAnyComplex& operator/= ( cuAnyComplex& z, real a ) {
    z.x /= a;
    z.y /= a;
    return z;
}

class cudaRandomWalker {
    friend __global__ void DoWorkKernel( cudaRandomWalker* walkers, real* S_mn, int device_id );
public:
    __device__ void Init( std::int64_t N_pos, std::int64_t N_neg ) {
        this->N_pos = N_pos;
        this->N_neg = N_neg;
        alpha = 0;
        curand_init( blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x, 0, &gen );
    }

    __device__ int Advance() {
        int sigma = RandomSign();
        (sigma == 1 ? N_pos : N_neg)--;
        alpha += sigma;
        return sigma;
    }

    __device__ std::int64_t get_alpha() const { return alpha; }

private:
    __device__ int RandomSign() {
        return (curand_uniform_real(&gen) * real(N_pos + N_neg) <= N_neg) ? -1 : 1;
    }

    std::int64_t N_pos;
    std::int64_t N_neg;
    std::int64_t alpha; // alpha[i]
    curandState_t gen;
};

static __global__ void DoWorkKernel( cudaRandomWalker* walkers, std::int64_t* ns, std::int64_t* ms,
                                     std::int64_t* N_poss, std::int64_t* N_negs, real* betas, /*OUT*/ real* Ss, /*OUT*/ real* o_os )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //int tid = ((blockI blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    cudaRandomWalker& walker = walkers[tid];
    std::int64_t n = ns[tid];
    std::int64_t m = ms[tid];
    std::int64_t N_pos = N_poss[tid];
    std::int64_t N_neg = N_negs[tid];
    real beta = betas[tid];
    real& S = Ss[tid];
    real& o_o = o_os[tid];

    assert(n < m);
    std::int64_t M = N_pos + N_neg;
    int sigma_n, sigma_m, alpha_m, alpha_n;
    cuAnyComplex S_nm = make_cuComplex(0, 0);
    cuAnyComplex S_mn = make_cuComplex(0, 0);

    walker.Init( N_pos, N_neg );
    std::int64_t i = 0;
    for ( ; i != n; i++ ) { // i = [0; n)
        S_mn += expi( walker.get_alpha() * beta );
        walker.Advance();
    }

    alpha_n = walker.get_alpha();
    S_nm += expi( alpha_n * beta );
    sigma_n = walker.Advance(); // i = n
    for ( i++; i != m; i++ ) { // i = (n, m)
        S_nm += expi( walker.get_alpha() * beta );
        walker.Advance();
    }

    alpha_m = walker.get_alpha();
    S_mn += expi(alpha_m * beta);
    sigma_m = walker.Advance(); // i = m
    for ( i++; i != M; i++ ) { // i = (m, M)
        S_mn += expi(walker.get_alpha() * beta);
        walker.Advance();
    }

    /*
    -\frac{1}{2} \cot ^2\left(\frac{\beta }{2}\right) \sigma _m \sigma _n \sin ^2\left(\frac{1}{4} \left(2 \alpha _m+\beta  \left(\sigma _m-\sigma _n\right)-2 \alpha _n\right)\right)
    */
    o_o = -M * (M - 1) / 2 * sigma_n * sigma_m / (2 * pow(tan(beta / 2), real(2.0))) * pow( sin(beta / 4 * (2 * (alpha_m - alpha_n) + sigma_m - sigma_n)), real(2.0) );

    S_nm /= real(m - n);
    S_mn /= real(n + M - m);
    S = cuCabsr((S_nm - S_mn) / (2 * sin(beta / 2)));
}

template <class T>
struct pair_ptr {
    T* host_ptr;
    T* device_ptr;

    void allocate( size_t size ) {
        host_ptr = new T[size];
        CheckErrorCode( cudaMalloc( (void**)&device_ptr, size * sizeof(T) ) );
    }

    ~pair_ptr() {
        CheckErrorCode( cudaFree(device_ptr) );
        delete[] host_ptr;
    }

    void CopyToDevice( size_t size ) {
        CheckErrorCode( cudaMemcpy( device_ptr, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice ) );
    }

    void CopyFromDevice( size_t size ) {
        CheckErrorCode( cudaMemcpy( host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost ) );
    }
};

inline void gpuAssert( cudaError_t code, const char *file, int line, bool abort=false ) {
    if ( code != cudaSuccess ) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define chkWrap(fn) \
template <typename... T>\
__host__ void chk##fn( T&&... args ) {\
    gpuErrchk( fn( std::forward<T>(args)... ) );\
}

chkWrap(cudaEventCreate)
chkWrap(cudaEventDestroy)
chkWrap(cudaEventElapsedTime)
chkWrap(cudaEventRecord)
chkWrap(cudaGetDeviceProperties)

double benchmark( int gridSize, int nThreads, int device ) {
    const int totThreads = gridSize * nThreads;
    CheckErrorCode( cudaSetDevice(device) );

    pair_ptr<cudaRandomWalker> walkers;
    pair_ptr<std::int64_t> ns, ms, N_poss, N_negs;
    pair_ptr<real> betas, Ss, o_os;

    walkers.allocate(totThreads);
    ns.allocate(totThreads);
    ms.allocate(totThreads);
    N_poss.allocate(totThreads);
    N_negs.allocate(totThreads);
    betas.allocate(totThreads);
    Ss.allocate(totThreads);
    o_os.allocate(totThreads);

    std::mt19937_64 gen;
    std::int64_t M = 1 << 15;
    cudaDeviceProp devProp;
    chkcudaGetDeviceProperties( &devProp, device );
    const int warpSize = devProp.warpSize;
    if ( totThreads < warpSize ) { return -1; }
    for ( size_t i = 0; i != totThreads / warpSize; i++ ) {
        std::uniform_int_distribution<std::int64_t> unif_M( 1, M );
        std::uniform_int_distribution<std::int64_t> unif_M1( 1, M - 1 );
        std::int64_t n = unif_M(gen) - 1;
        std::int64_t m = (n + unif_M1(gen)) % M;
        if ( n > m ) {
            std::swap( n, m );
        }
        for ( int j = 0; j != warpSize; j++ ) {
            size_t idx = i * warpSize + j;
            ns.host_ptr[idx] = n;
            ms.host_ptr[idx] = m;
            N_poss.host_ptr[idx] = M / 2;
            N_negs.host_ptr[idx] = M / 2;
            betas.host_ptr[idx] = real(0.1);
        }
    }

    walkers.CopyToDevice(totThreads);
    ns.CopyToDevice(totThreads);
    ms.CopyToDevice(totThreads);
    N_poss.CopyToDevice(totThreads);
    N_negs.CopyToDevice(totThreads);
    betas.CopyToDevice(totThreads);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    chkcudaEventCreate( &start );
    chkcudaEventCreate( &stop );
    chkcudaEventRecord( start, cudaStream_t(0) );

    DoWorkKernel<<<gridSize, nThreads>>>( walkers.device_ptr, ns.device_ptr, ms.device_ptr, N_poss.device_ptr, N_negs.device_ptr,
                                         betas.device_ptr, Ss.device_ptr, o_os.device_ptr );

    if ( cudaGetLastError() != cudaSuccess ) { return -1; }
    if ( cudaDeviceSynchronize() != cudaSuccess ) { return -1; }
    chkcudaEventRecord( stop, cudaStream_t(0) );
    if ( cudaEventSynchronize(stop) != cudaSuccess ) { return -1; }

    chkcudaEventElapsedTime( &gpuTime, start, stop );

    double speed = totThreads * M / gpuTime * 1e3;

    chkcudaEventDestroy(start);
    chkcudaEventDestroy(stop);

    Ss.CopyFromDevice(totThreads);
    o_os.CopyFromDevice(totThreads);

    //for ( size_t i = 0; i != std::min(30, totThreads); i++ ) {
    //    std::cout << Ss.host_ptr[i] << '\t' << o_os.host_ptr[i] << std::endl;
    //}
    return speed;
}

int main() {
    int device = 0;
    int best_gridSize = 1;
    int best_nThreads = 1;
    double best_speed = 1;
    std::cout << "gridSize" << '\t' << "nThreads" << '\t' << "speed" << std::endl;
    cudaDeviceProp devProp;
    chkcudaGetDeviceProperties( &devProp, device );
    std::cout << "Running on GPU \"" << devProp.name << "\"" << std::endl;
    if ( devProp.name == "NVIDIA GeForce GTX 1080 Ti"s ) {
        device = 1;
    }
    const int warpSize = devProp.warpSize;

    for ( int gridSize = 1; gridSize <= 1 << 12; gridSize *= 2 )
    //int gridSize = 4096;
    {
        for ( int nThreads = 1; nThreads <= 512; nThreads *= 2 )
        //int nThreads = 128;
        {
            if ( gridSize * nThreads < warpSize ) { continue; }
            double speed = benchmark( gridSize, nThreads, device );
            if ( speed == -1 ) {
                std::cout << "Fail for " << gridSize << '\t' << nThreads << std::endl;
            } else {
                std::cout << gridSize << '\t' << nThreads << '\t' << speed;
                if ( speed > best_speed ) {
                    best_speed = speed;
                    best_gridSize = gridSize;
                    best_nThreads = nThreads;
                    std::cout << "\t(new best)";
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << "--------------------------------------------\n";
    std::cout << "The best is " << best_gridSize << "x" << best_nThreads << " giving " << best_speed << " speed\n";
    return 0;
}
