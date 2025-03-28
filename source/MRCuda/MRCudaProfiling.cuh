#pragma once

#include "MRCuda.cuh"

#include "MRMesh/MRTimer.h"
#include "MRPch/MRSpdlog.h"

#include <cassert>
#include <numeric>

namespace MR::Cuda
{

struct ProfilingSettings
{
    size_t dynSharedMemSize{ 0 };
    cudaStream_t stream{ nullptr };
    int deviceId{ -1 };
};

/// test kernel run duration for different launch parameters
template <typename Kernel, typename... Args>
void runProfiling( size_t size, const ProfilingSettings& settings, Kernel&& kernel, Args&&... args )
{
    assert( size > 0 );

    auto deviceId = settings.deviceId;
    if ( deviceId < 0 )
        cudaGetDevice( &deviceId );

    cudaDeviceProp props;
    cudaGetDeviceProperties( &props, deviceId );

    const auto maxWarpsPerBlock = props.maxThreadsPerBlock / props.warpSize;
    const auto maxWarpsPerSM = props.maxThreadsPerMultiProcessor / props.warpSize;
    assert( props.warpSize * maxWarpsPerBlock == props.maxThreadsPerBlock );
    assert( props.warpSize * maxWarpsPerSM == props.maxThreadsPerMultiProcessor );
    const auto optimalNumThreadsPerBlock = props.warpSize * std::gcd( maxWarpsPerBlock, maxWarpsPerSM );
    spdlog::info( "optimal number of threads per block: {}", optimalNumThreadsPerBlock );

    struct Record
    {
        int threadsPerBlock;
        int maxActiveBlocksPerSM;
        int maxActiveBlocks;
        int maxActiveThreads;
        size_t durationMs;
    };
    std::vector<Record> records;
    for ( auto threadsPerBlock = props.warpSize; threadsPerBlock <= props.maxThreadsPerBlock; threadsPerBlock += props.warpSize )
    {
        Record rec { .threadsPerBlock = threadsPerBlock };
        cudaOccupancyMaxActiveBlocksPerMultiprocessor( &rec.maxActiveBlocksPerSM, kernel, threadsPerBlock, settings.dynSharedMemSize );
        rec.maxActiveBlocks = rec.maxActiveBlocksPerSM * props.multiProcessorCount;
        rec.maxActiveThreads = threadsPerBlock * rec.maxActiveBlocks;
        spdlog::info( "-----" );
        spdlog::info( "threads per block: {}", threadsPerBlock );
        spdlog::info( "max active blocks per multi processor: {}", rec.maxActiveBlocksPerSM );
        spdlog::info( "max active blocks: {}", rec.maxActiveBlocks );
        spdlog::info( "max active threads: {}", rec.maxActiveThreads );

        const auto numBlocks = (unsigned int)( ( size + threadsPerBlock - 1 ) / threadsPerBlock );
        const auto dynSharedMemSizePerBlock = settings.dynSharedMemSize / numBlocks;

        MR::Timer timer( "kernel profiling" );
        kernel <<< numBlocks, threadsPerBlock, dynSharedMemSizePerBlock, settings.stream >>> ( std::forward<Args>( args )... );
        cudaDeviceSynchronize();
        rec.durationMs = std::chrono::duration_cast<std::chrono::milliseconds>( timer.secondsPassed() ).count();
        spdlog::info( "kernel time: {} ms", rec.durationMs );

        records.emplace_back( rec );
    }

    std::stringstream ss;
    ss << "-----\n";
    for ( const auto& rec : records )
        ss << fmt::format( "{},{},{},{},{}\n", rec.threadsPerBlock, rec.maxActiveBlocksPerSM, rec.maxActiveBlocks, rec.maxActiveThreads, rec.durationMs );
    spdlog::info( ss.str() );
}

} // namespace MR::Cuda
