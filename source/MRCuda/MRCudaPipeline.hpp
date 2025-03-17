#pragma once

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRFinally.h"
#include "MRPch/MRTBB.h"

namespace MR::Cuda
{

template <typename BufferType, typename InputIt, typename GPUFunc, typename CPUFunc>
Expected<void> cudaPipeline( BufferType init, InputIt begin, InputIt end, GPUFunc gpuFunc, CPUFunc cpuFunc )
{
    std::array<BufferType, 2> buffers { init, init };
    std::array<InputIt, 2> it;
    enum Device
    {
        GPU = 0,
        CPU = 1,
    };

    tbb::task_group gpuTask;
    MR_FINALLY {
        gpuTask.wait();
    };

    for ( it[GPU] = begin; it[GPU] != end; it[CPU] = it[GPU]++ )
    {
        // TODO: replace with cudaStream usage
        Expected<void> gpuRes;
        gpuTask.run( [&]
        {
            gpuRes = gpuFunc( buffers[GPU], *it[GPU] );
        } );

        if ( it[GPU] != begin )
        {
            if ( auto cpuRes = cpuFunc( buffers[CPU], *it[CPU] ); !cpuRes )
                return cpuRes;
        }

        gpuTask.wait();
        if ( !gpuRes )
            return gpuRes;

        std::swap( buffers[GPU], buffers[CPU] );
    }
    // process the last item
    return cpuFunc( buffers[CPU], *it[CPU] );
}

} // namespace MR::Cuda
