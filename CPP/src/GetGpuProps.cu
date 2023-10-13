#include <stdio.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h> // for Intellisense

template <bool b>
struct f_helper;

template <>
struct f_helper<true> {
    void print( std::ostream& stream, const char* ptr ) {
        stream << ptr;
    }
};

template <>
struct f_helper<false> {
    template <std::size_t N>
    void print( std::ostream& stream, const char (&ar)[N] ) {
        if ( N >= 256 && ar[N - 1] == '\0' ) {
            stream << ar;
        } else {
            std::ios_base::fmtflags oldFlags = stream.flags();
            std::streamsize oldWidth = stream.width();
            char oldFill = stream.fill();
            for ( size_t i = 0; i != N; i++ ) {
                if ( i != 0 ) {
                    stream << '-';
                }
                stream << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << (unsigned int)(unsigned char)ar[i];
            }
            stream.flags(oldFlags);
            stream << std::setw(oldWidth) << std::setfill(oldFill);
        }
    }
};

template <class T>
auto printval( std::ostream& stream, T&& value ) -> typename std::enable_if<std::is_same<char*, std::decay_t<T>>::value ||
                                                                        std::is_same<const char*, std::decay_t<T>>::value, void>::type
{
    f_helper<std::is_pointer<std::remove_reference_t<T>>::value>{}.print( stream, value );
}

template <class T>
auto printval( std::ostream& stream, T&& value ) -> typename std::enable_if<!std::is_same<char*, std::decay_t<T>>::value &&
                                                                        !std::is_same<const char*, std::decay_t<T>>::value, void>::type
{
    std::cout << value;
}

std::ostream& operator<< ( std::ostream& stream, const cudaUUID_t& uuid ) {
    printval( stream, uuid.bytes );
    return stream;
}

template <std::size_t N>
std::ostream& operator<< ( std::ostream& stream, const int (&ar)[N] ) {
    stream << '[';
    for ( size_t i = 0; i != N; i++ ) {
        if ( i != 0 ) {
            stream << ", ";
        }
        stream << (unsigned int)(unsigned char)ar[i];
    }
    return stream << ']';
}

template <class T>
void printField( T&& value, const char* comment ) {
    std::cout << std::setw(40) << std::left << comment << ": ";
    printval( std::cout, value );
    std::cout << '\n';
}

#define print( devProp, field, comment )                               \
    {                                                                  \
        for ( size_t i = 0; i != devProp.size(); i++ ) {               \
            printField( (devProp)[i].field, i == 0 ? (comment) : "" ); \
        }                                                              \
    }

int main ( int argc, char * argv [] ) {
    int            deviceCount;

    cudaGetDeviceCount ( &deviceCount );
    std::vector<cudaDeviceProp> devProp(deviceCount);

    printf ( "Found %d devices\n", deviceCount );

    for ( int device = 0; device < deviceCount; device++ ) {
        cudaGetDeviceProperties( &devProp[device], device );
    }

    //printf( "Device %d\n", device );
    //std::cout << "Compute capability     : " << devProp.major << "." << devProp.minor << "\n";
    //std::cout << "Name                   : " << devProp.name << "\n";
    //printf( "Total Global Memory    : %zu\n", devProp.totalGlobalMem );
    //printf( "Shared memory per block: %d\n", devProp.sharedMemPerBlock );
    //printf( "Registers per block    : %d\n", devProp.regsPerBlock );
    //printf( "Warp size              : %d\n", devProp.warpSize );
    //printf( "Max threads per block  : %d\n", devProp.maxThreadsPerBlock);
    //printf( "Total constant memory  : %d\n", devProp.totalConstMem);
    //printf( "Clock Rate             : %d\n", devProp.clockRate);
    //printf( "Texture Alignment      : %u\n", devProp.textureAlignment);
    //printf( "Device Overlap         : %d\n", devProp.deviceOverlap);
    //printf( "Multiprocessor Count   : %d\n", devProp.multiProcessorCount );
    //printf( "Max Threads Dim        : %d,%d,%d\n", devProp.maxThreadsDim[0],
    //                                         devProp.maxThreadsDim[1],
    //                                         devProp.maxThreadsDim[2] );
    //printf( "Max Grid Size          : %d,%d,%d\n", devProp.maxGridSize[0],
    //                                         devProp.maxGridSize[1],
    //                                         devProp.maxGridSize[2] );

    print( devProp, name                                  , "ASCII string identifying device" );
    print( devProp, uuid                                  , "16-byte unique identifier" );
    print( devProp, luid                                  , "8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms" );
    print( devProp, luidDeviceNodeMask                    , "LUID device node mask. Value is undefined on TCC and non-Windows platforms" );
    print( devProp, totalGlobalMem                        , "Global memory available on device in bytes" );
    print( devProp, sharedMemPerBlock                     , "Shared memory available per block in bytes" );
    print( devProp, regsPerBlock                          , "32-bit registers available per block" );
    print( devProp, warpSize                              , "Warp size in threads" );
    print( devProp, memPitch                              , "Maximum pitch in bytes allowed by memory copies" );
    print( devProp, maxThreadsPerBlock                    , "Maximum number of threads per block" );
    print( devProp, maxThreadsDim                         , "Maximum size of each dimension of a block" );
    print( devProp, maxGridSize                           , "Maximum size of each dimension of a grid" );
    print( devProp, clockRate                             , "Clock frequency in kilohertz" );
    print( devProp, totalConstMem                         , "Constant memory available on device in bytes" );
    print( devProp, major                                 , "Major compute capability" );
    print( devProp, minor                                 , "Minor compute capability" );
    print( devProp, textureAlignment                      , "Alignment requirement for textures" );
    print( devProp, texturePitchAlignment                 , "Pitch alignment requirement for texture references bound to pitched memory" );
    print( devProp, deviceOverlap                         , "Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount." );
    print( devProp, multiProcessorCount                   , "Number of multiprocessors on device" );
    print( devProp, kernelExecTimeoutEnabled              , "Specified whether there is a run time limit on kernels" );
    print( devProp, integrated                            , "Device is integrated as opposed to discrete" );
    print( devProp, canMapHostMemory                      , "Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer" );
    print( devProp, computeMode                           , "Compute mode (See ::cudaComputeMode)" );
    print( devProp, maxTexture1D                          , "Maximum 1D texture size" );
    print( devProp, maxTexture1DMipmap                    , "Maximum 1D mipmapped texture size" );
    print( devProp, maxTexture1DLinear                    , "Maximum size for 1D textures bound to linear memory" );
    print( devProp, maxTexture2D                          , "Maximum 2D texture dimensions" );
    print( devProp, maxTexture2DMipmap                    , "Maximum 2D mipmapped texture dimensions" );
    print( devProp, maxTexture2DLinear                    , "Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory" );
    print( devProp, maxTexture2DGather                    , "Maximum 2D texture dimensions if texture gather operations have to be performed" );
    print( devProp, maxTexture3D                          , "Maximum 3D texture dimensions" );
    print( devProp, maxTexture3DAlt                       , "Maximum alternate 3D texture dimensions" );
    print( devProp, maxTextureCubemap                     , "Maximum Cubemap texture dimensions" );
    print( devProp, maxTexture1DLayered                   , "Maximum 1D layered texture dimensions" );
    print( devProp, maxTexture2DLayered                   , "Maximum 2D layered texture dimensions" );
    print( devProp, maxTextureCubemapLayered              , "Maximum Cubemap layered texture dimensions" );
    print( devProp, maxSurface1D                          , "Maximum 1D surface size" );
    print( devProp, maxSurface2D                          , "Maximum 2D surface dimensions" );
    print( devProp, maxSurface3D                          , "Maximum 3D surface dimensions" );
    print( devProp, maxSurface1DLayered                   , "Maximum 1D layered surface dimensions" );
    print( devProp, maxSurface2DLayered                   , "Maximum 2D layered surface dimensions" );
    print( devProp, maxSurfaceCubemap                     , "Maximum Cubemap surface dimensions" );
    print( devProp, maxSurfaceCubemapLayered              , "Maximum Cubemap layered surface dimensions" );
    print( devProp, surfaceAlignment                      , "Alignment requirements for surfaces" );
    print( devProp, concurrentKernels                     , "Device can possibly execute multiple kernels concurrently" );
    print( devProp, ECCEnabled                            , "Device has ECC support enabled" );
    print( devProp, pciBusID                              , "PCI bus ID of the device" );
    print( devProp, pciDeviceID                           , "PCI device ID of the device" );
    print( devProp, pciDomainID                           , "PCI domain ID of the device" );
    print( devProp, tccDriver                             , "1 if device is a Tesla device using TCC driver, 0 otherwise" );
    print( devProp, asyncEngineCount                      , "Number of asynchronous engines" );
    print( devProp, unifiedAddressing                     , "Device shares a unified address space with the host" );
    print( devProp, memoryClockRate                       , "Peak memory clock frequency in kilohertz" );
    print( devProp, memoryBusWidth                        , "Global memory bus width in bits" );
    print( devProp, l2CacheSize                           , "Size of L2 cache in bytes" );
    print( devProp, maxThreadsPerMultiProcessor           , "Maximum resident threads per multiprocessor" );
    print( devProp, streamPrioritiesSupported             , "Device supports stream priorities" );
    print( devProp, globalL1CacheSupported                , "Device supports caching globals in L1" );
    print( devProp, localL1CacheSupported                 , "Device supports caching locals in L1" );
    print( devProp, sharedMemPerMultiprocessor            , "Shared memory available per multiprocessor in bytes" );
    print( devProp, regsPerMultiprocessor                 , "32-bit registers available per multiprocessor" );
    print( devProp, managedMemory                         , "Device supports allocating managed memory on this system" );
    print( devProp, isMultiGpuBoard                       , "Device is on a multi-GPU board" );
    print( devProp, multiGpuBoardGroupID                  , "Unique identifier for a group of devices on the same multi-GPU board" );
    print( devProp, hostNativeAtomicSupported             , "Link between the device and the host supports native atomic operations" );
    print( devProp, singleToDoublePrecisionPerfRatio      , "Ratio of single precision performance (in floating-point operations per second) to double precision performance" );
    print( devProp, pageableMemoryAccess                  , "Device supports coherently accessing pageable memory without calling cudaHostRegister on it" );
    print( devProp, concurrentManagedAccess               , "Device can coherently access managed memory concurrently with the CPU" );
    print( devProp, computePreemptionSupported            , "Device supports Compute Preemption" );
    print( devProp, canUseHostPointerForRegisteredMem     , "Device can access host registered memory at the same virtual address as the CPU" );
    print( devProp, cooperativeLaunch                     , "Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel" );
    print( devProp, cooperativeMultiDeviceLaunch          , "Device can participate in cooperative kernels launched via ::cudaLaunchCooperativeKernelMultiDevice" );
    print( devProp, sharedMemPerBlockOptin                , "Per device maximum shared memory per block usable by special opt in" );
    print( devProp, pageableMemoryAccessUsesHostPageTables, "Device accesses pageable memory via the host's page tables" );
    print( devProp, directManagedMemAccessFromHost        , "Host can directly access managed memory on the device without migration." );

    return 0;
}
