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

#define print( devProp, field, comment )  \
    printField( devProp.field, comment ); \

int main ( int argc, char * argv [] ) {
    int            deviceCount;

    cudaGetDeviceCount ( &deviceCount );
    std::vector<cudaDeviceProp> devProp(deviceCount);

    printf ( "Found %d devices\n", deviceCount );
#if 0
    for ( int device = 0; device < deviceCount; device++ ) {
        cudaGetDeviceProperties( &devProp[device], device );

        printf( "\nDevice %d\n", device );
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

        print( devProp[device], name                                  , "ASCII string identifying device" );
        print( devProp[device], uuid                                  , "16-byte unique identifier" );
        print( devProp[device], luid                                  , "8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms" );
        print( devProp[device], luidDeviceNodeMask                    , "LUID device node mask. Value is undefined on TCC and non-Windows platforms" );
        print( devProp[device], totalGlobalMem                        , "Global memory available on device in bytes" );
        print( devProp[device], sharedMemPerBlock                     , "Shared memory available per block in bytes" );
        print( devProp[device], regsPerBlock                          , "32-bit registers available per block" );
        print( devProp[device], warpSize                              , "Warp size in threads" );
        print( devProp[device], memPitch                              , "Maximum pitch in bytes allowed by memory copies" );
        print( devProp[device], maxThreadsPerBlock                    , "Maximum number of threads per block" );
        print( devProp[device], maxThreadsDim                         , "Maximum size of each dimension of a block" );
        print( devProp[device], maxGridSize                           , "Maximum size of each dimension of a grid" );
        print( devProp[device], clockRate                             , "Clock frequency in kilohertz" );
        print( devProp[device], totalConstMem                         , "Constant memory available on device in bytes" );
        print( devProp[device], major                                 , "Major compute capability" );
        print( devProp[device], minor                                 , "Minor compute capability" );
        print( devProp[device], textureAlignment                      , "Alignment requirement for textures" );
        print( devProp[device], texturePitchAlignment                 , "Pitch alignment requirement for texture references bound to pitched memory" );
        print( devProp[device], deviceOverlap                         , "Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount." );
        print( devProp[device], multiProcessorCount                   , "Number of multiprocessors on device" );
        print( devProp[device], kernelExecTimeoutEnabled              , "Specified whether there is a run time limit on kernels" );
        print( devProp[device], integrated                            , "Device is integrated as opposed to discrete" );
        print( devProp[device], canMapHostMemory                      , "Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer" );
        print( devProp[device], computeMode                           , "Compute mode (See ::cudaComputeMode)" );
        print( devProp[device], maxTexture1D                          , "Maximum 1D texture size" );
        print( devProp[device], maxTexture1DMipmap                    , "Maximum 1D mipmapped texture size" );
        print( devProp[device], maxTexture1DLinear                    , "Maximum size for 1D textures bound to linear memory" );
        print( devProp[device], maxTexture2D                          , "Maximum 2D texture dimensions" );
        print( devProp[device], maxTexture2DMipmap                    , "Maximum 2D mipmapped texture dimensions" );
        print( devProp[device], maxTexture2DLinear                    , "Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory" );
        print( devProp[device], maxTexture2DGather                    , "Maximum 2D texture dimensions if texture gather operations have to be performed" );
        print( devProp[device], maxTexture3D                          , "Maximum 3D texture dimensions" );
        print( devProp[device], maxTexture3DAlt                       , "Maximum alternate 3D texture dimensions" );
        print( devProp[device], maxTextureCubemap                     , "Maximum Cubemap texture dimensions" );
        print( devProp[device], maxTexture1DLayered                   , "Maximum 1D layered texture dimensions" );
        print( devProp[device], maxTexture2DLayered                   , "Maximum 2D layered texture dimensions" );
        print( devProp[device], maxTextureCubemapLayered              , "Maximum Cubemap layered texture dimensions" );
        print( devProp[device], maxSurface1D                          , "Maximum 1D surface size" );
        print( devProp[device], maxSurface2D                          , "Maximum 2D surface dimensions" );
        print( devProp[device], maxSurface3D                          , "Maximum 3D surface dimensions" );
        print( devProp[device], maxSurface1DLayered                   , "Maximum 1D layered surface dimensions" );
        print( devProp[device], maxSurface2DLayered                   , "Maximum 2D layered surface dimensions" );
        print( devProp[device], maxSurfaceCubemap                     , "Maximum Cubemap surface dimensions" );
        print( devProp[device], maxSurfaceCubemapLayered              , "Maximum Cubemap layered surface dimensions" );
        print( devProp[device], surfaceAlignment                      , "Alignment requirements for surfaces" );
        print( devProp[device], concurrentKernels                     , "Device can possibly execute multiple kernels concurrently" );
        print( devProp[device], ECCEnabled                            , "Device has ECC support enabled" );
        print( devProp[device], pciBusID                              , "PCI bus ID of the device" );
        print( devProp[device], pciDeviceID                           , "PCI device ID of the device" );
        print( devProp[device], pciDomainID                           , "PCI domain ID of the device" );
        print( devProp[device], tccDriver                             , "1 if device is a Tesla device using TCC driver, 0 otherwise" );
        print( devProp[device], asyncEngineCount                      , "Number of asynchronous engines" );
        print( devProp[device], unifiedAddressing                     , "Device shares a unified address space with the host" );
        print( devProp[device], memoryClockRate                       , "Peak memory clock frequency in kilohertz" );
        print( devProp[device], memoryBusWidth                        , "Global memory bus width in bits" );
        print( devProp[device], l2CacheSize                           , "Size of L2 cache in bytes" );
        print( devProp[device], maxThreadsPerMultiProcessor           , "Maximum resident threads per multiprocessor" );
        print( devProp[device], streamPrioritiesSupported             , "Device supports stream priorities" );
        print( devProp[device], globalL1CacheSupported                , "Device supports caching globals in L1" );
        print( devProp[device], localL1CacheSupported                 , "Device supports caching locals in L1" );
        print( devProp[device], sharedMemPerMultiprocessor            , "Shared memory available per multiprocessor in bytes" );
        print( devProp[device], regsPerMultiprocessor                 , "32-bit registers available per multiprocessor" );
        print( devProp[device], managedMemory                         , "Device supports allocating managed memory on this system" );
        print( devProp[device], isMultiGpuBoard                       , "Device is on a multi-GPU board" );
        print( devProp[device], multiGpuBoardGroupID                  , "Unique identifier for a group of devices on the same multi-GPU board" );
        print( devProp[device], hostNativeAtomicSupported             , "Link between the device and the host supports native atomic operations" );
        print( devProp[device], singleToDoublePrecisionPerfRatio      , "Ratio of single precision performance (in floating-point operations per second) to double precision performance" );
        print( devProp[device], pageableMemoryAccess                  , "Device supports coherently accessing pageable memory without calling cudaHostRegister on it" );
        print( devProp[device], concurrentManagedAccess               , "Device can coherently access managed memory concurrently with the CPU" );
        print( devProp[device], computePreemptionSupported            , "Device supports Compute Preemption" );
        print( devProp[device], canUseHostPointerForRegisteredMem     , "Device can access host registered memory at the same virtual address as the CPU" );
        print( devProp[device], cooperativeLaunch                     , "Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel" );
        print( devProp[device], cooperativeMultiDeviceLaunch          , "Device can participate in cooperative kernels launched via ::cudaLaunchCooperativeKernelMultiDevice" );
        print( devProp[device], sharedMemPerBlockOptin                , "Per device maximum shared memory per block usable by special opt in" );
        print( devProp[device], pageableMemoryAccessUsesHostPageTables, "Device accesses pageable memory via the host's page tables" );
        print( devProp[device], directManagedMemAccessFromHost        , "Host can directly access managed memory on the device without migration." );
    }
#endif
    return 0;
}
