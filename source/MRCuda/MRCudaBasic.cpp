#include "MRCudaBasic.h"
#include "MRCudaBasic.hpp"

#include "MRCuda.cuh"

#include <MRMesh/MRVector2.h>
#include <MRMesh/MRVector3.h>
#include <MRPch/MRSpdlog.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace MR
{

namespace Cuda
{

namespace
{

/// https://en.wikipedia.org/wiki/CUDA Compute Capability (CUDA SDK support vs. Microarchitecture)
bool computeTooOldForRuntime( int runtimeVersion, int computeMajor, int computeMinor )
{
    if ( runtimeVersion / 1000 >= 12 && computeMajor < 5 )
        return true;
    if ( runtimeVersion / 1000 > 10 && ( computeMajor < 3 || ( computeMajor == 3 && computeMinor < 5 ) ) )
        return true;
    return false;
}

struct DriverDevice
{
    int computeMajor = 0;
    int computeMinor = 0;
    /// maximum CUDA version the driver supports, as the driver itself reports it
    int driverVersion = 0;
    std::string name;
};

/// Asks the driver itself about the most capable device present, bypassing the
/// CUDA runtime.
/// cudaGetDeviceCount() refuses outright when the driver predates the runtime, so
/// the runtime cannot tell a merely out-of-date driver from a card that CUDA no
/// longer supports at all; the driver API still answers in both cases.
/// The library ships with the NVIDIA driver, so this fails on HIP/AMD builds or
/// when no driver is installed, and the caller keeps the runtime error.
Expected<DriverDevice> queryDriverApi()
{
#ifdef _WIN32
    const char * libName = "nvcuda.dll";
    const auto lib = LoadLibraryA( libName );
    const auto libError = [] { return std::to_string( GetLastError() ); };
#else
    const char * libName = "libcuda.so.1";
    const auto lib = dlopen( libName, RTLD_LAZY );
    const auto libError = [] { const char * e = dlerror(); return std::string( e ? e : "unknown" ); };
#endif
    if ( !lib )
        return MR::unexpected( fmt::format( "cannot load {}: {}", libName, libError() ) );

    // the driver stays loaded for the process either way, but do not add a
    // reference on every call: getDeviceInfo() is not called only once
    struct LibGuard
    {
        decltype( lib ) h;
        ~LibGuard()
        {
#ifdef _WIN32
            FreeLibrary( h );
#else
            dlclose( h );
#endif
        }
    } libGuard{ lib };

    auto sym = [lib] ( const char * name )
    {
#ifdef _WIN32
        return (void *)GetProcAddress( lib, name );
#else
        return dlsym( lib, name );
#endif
    };

    // declared here rather than via <cuda.h>: that header is absent in HIP builds,
    // and these entry points have never been versioned
    const auto cuInit = (int (*)( unsigned ))sym( "cuInit" );
    const auto cuDriverGetVersion = (int (*)( int * ))sym( "cuDriverGetVersion" );
    const auto cuDeviceGetCount = (int (*)( int * ))sym( "cuDeviceGetCount" );
    const auto cuDeviceGet = (int (*)( int *, int ))sym( "cuDeviceGet" );
    const auto cuDeviceGetAttribute = (int (*)( int *, int, int ))sym( "cuDeviceGetAttribute" );
    const auto cuDeviceGetName = (int (*)( char *, int, int ))sym( "cuDeviceGetName" );
    if ( !cuInit || !cuDriverGetVersion || !cuDeviceGetCount || !cuDeviceGet ||
         !cuDeviceGetAttribute || !cuDeviceGetName )
        return MR::unexpected( fmt::format( "{} misses an expected entry point", libName ) );

    constexpr int cCudaSuccess = 0;
    constexpr int cAttrComputeMajor = 75; // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    constexpr int cAttrComputeMinor = 76; // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

    if ( const auto code = cuInit( 0 ); code != cCudaSuccess )
        return MR::unexpected( fmt::format( "cuInit failed with code {}", code ) );

    DriverDevice res;
    if ( const auto code = cuDriverGetVersion( &res.driverVersion ); code != cCudaSuccess )
        return MR::unexpected( fmt::format( "cuDriverGetVersion failed with code {}", code ) );

    int devCount = 0;
    if ( const auto code = cuDeviceGetCount( &devCount ); code != cCudaSuccess )
        return MR::unexpected( fmt::format( "cuDeviceGetCount failed with code {}", code ) );
    if ( devCount <= 0 )
        return MR::unexpected( "the driver reports no CUDA devices" );

    // the most capable device decides: saying a card is unsupported is only true
    // of the machine if every card in it is
    int best = -1;
    for ( int i = 0; i < devCount; ++i )
    {
        int dev = 0, major = 0, minor = 0;
        if ( cuDeviceGet( &dev, i ) != cCudaSuccess ||
             cuDeviceGetAttribute( &major, cAttrComputeMajor, dev ) != cCudaSuccess ||
             cuDeviceGetAttribute( &minor, cAttrComputeMinor, dev ) != cCudaSuccess )
            continue;
        if ( major < res.computeMajor || ( major == res.computeMajor && minor <= res.computeMinor ) )
            continue;
        res.computeMajor = major;
        res.computeMinor = minor;
        best = dev;
    }
    if ( best < 0 )
        return MR::unexpected( fmt::format( "no compute capability could be read from {} device(s)", devCount ) );

    char name[256] = {};
    if ( cuDeviceGetName( name, (int)sizeof( name ) - 1, best ) == cCudaSuccess )
        res.name = name;
    return res;
}

} //anonymous namespace

Expected<DeviceInfo> getDeviceInfo()
{
    const auto dev0 = queryDriverApi();
    DeviceInfo res;
    CUDA_RETURN_UNEXPECTED( cudaDriverGetVersion( &res.driverVersion ) );
    if ( res.driverVersion <= 0 )
        return MR::unexpected( "NVIDIA GPU error: no CUDA driver found" );

    {
        int n = 0;
        auto code = cudaGetDeviceCount( &n );
        if ( code != cudaSuccess || n <= 0 )
        {
            int runtimeVersion = 0;
            if ( cudaRuntimeGetVersion( &runtimeVersion ) != cudaSuccess )
                runtimeVersion = 0; // leaves computeTooOldForRuntime() false, so we fall through
            // the runtime blames the driver whatever the reason, so ask the driver
            // whether this card is supported at all before telling anyone to update
            const auto dev = queryDriverApi();
            if ( dev && computeTooOldForRuntime( runtimeVersion, dev->computeMajor, dev->computeMinor ) )
            {
                return MR::unexpected( fmt::format(
                    "NVIDIA GPU error: {} has compute capability {}.{}, dropped by CUDA {}; "
                    "no driver update will help, CUDA driver {}.{}",
                    dev->name.empty() ? "the GPU" : dev->name, dev->computeMajor, dev->computeMinor,
                    runtimeVersion / 1000,
                    dev->driverVersion / 1000, ( dev->driverVersion % 1000 ) / 10 ) );
            }
            auto err = ( code != cudaSuccess ) ? MR::Cuda::getError( code ) : "NVIDIA GPU error: no capable device found";
            err += fmt::format( ", CUDA driver {}.{}", res.driverVersion / 1000, ( res.driverVersion % 1000 ) / 10 );
            if ( !dev )
                err += fmt::format( "; compute capability unknown: {}", dev.error() );
            return MR::unexpected( err );
        }
    }

    CUDA_RETURN_UNEXPECTED( cudaRuntimeGetVersion( &res.runtimeVersion ) );

    cudaDeviceProp prop;
    CUDA_RETURN_UNEXPECTED( cudaGetDeviceProperties( &prop, 0 ) );
    res.computeMajor = prop.major;
    res.computeMinor = prop.minor;
    res.totalGlobalMem = prop.totalGlobalMem;
    res.name = prop.name;

    return res;
}

bool DeviceInfo::fitForComputations() const
{
    if ( computeTooOldForRuntime( runtimeVersion, computeMajor, computeMinor ) )
        return false;

    return runtimeVersion <= driverVersion;
}

bool isCudaAvailable( int* driverVersionOut, int* runtimeVersionOut, int* computeMajorOut, int* computeMinorOut )
{
     auto info = MR::Cuda::getDeviceInfo();
     if ( !info )
         return false;

    if ( driverVersionOut )
        *driverVersionOut = info->driverVersion;
    if ( runtimeVersionOut )
        *runtimeVersionOut = info->runtimeVersion;
    if ( computeMajorOut )
        *computeMajorOut = info->computeMajor;
    if ( computeMinorOut )
        *computeMinorOut = info->computeMinor;

    return info->fitForComputations();
}

size_t getCudaAvailableMemory()
{
    if ( CUDA_EXEC( cudaSetDevice( 0 ) ) != cudaSuccess )
        return 0;
    size_t memFree = 0, memTot = 0;
    if ( CUDA_EXEC( cudaMemGetInfo( &memFree, &memTot ) ) )
        return 0;
    // minus extra 128 MB
    return memFree - 128 * 1024 * 1024;
}

size_t getCudaSafeMemoryLimit()
{
    constexpr float cMaxGpuMemoryUsage = 0.80f;
    return size_t( (float)getCudaAvailableMemory() * cMaxGpuMemoryUsage );
}

size_t maxBufferSize( size_t availableBytes, size_t elementCount, size_t elementBytes )
{
    return std::min( availableBytes / elementBytes, elementCount );
}

size_t maxBufferSizeAlignedByBlock( size_t availableBytes, const Vector2i& blockDims, size_t elementBytes )
{
    const auto rowSize = (size_t)blockDims.x;
    return std::min( availableBytes / elementBytes / rowSize, (size_t)blockDims.y ) * rowSize;
}

size_t maxBufferSizeAlignedByBlock( size_t availableBytes, const Vector3i& blockDims, size_t elementBytes )
{
    const auto layerSize = (size_t)blockDims.x * blockDims.y;
    return std::min( availableBytes / elementBytes / layerSize, (size_t)blockDims.z ) * layerSize;
}

/// cudaGetErrorString/-Name return nullptr for an unrecognized code, and
/// formatting a null string pointer throws fmt::format_error
static const char * cudaStrOrUnknown( const char * str )
{
    return str ? str : "unknown error";
}

std::string getError( cudaError_t code )
{
    return fmt::format( "NVIDIA GPU error: {}", cudaStrOrUnknown( cudaGetErrorString( code ) ) );
}

cudaError_t logError( cudaError_t code, const char * file, int line )
{
    if ( code == cudaSuccess )
        return code;

    if ( file )
    {
        spdlog::error("CUDA error {}: {}. In file: {} Line: {}", 
            cudaStrOrUnknown( cudaGetErrorName( code ) ), cudaStrOrUnknown( cudaGetErrorString( code ) ), file, line );
    }
    else
    {
        spdlog::error( "CUDA error {}: {}",
            cudaStrOrUnknown( cudaGetErrorName( code ) ), cudaStrOrUnknown( cudaGetErrorString( code ) ) );
    }
    return code;
}

} //namespace Cuda

} //namespace MR
