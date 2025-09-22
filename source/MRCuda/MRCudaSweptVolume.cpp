#include "MRCudaSweptVolume.h"
#ifndef MRCUDA_NO_VOXELS
#include "MRCudaSweptVolume.cuh"

#include "MRCuda.cuh"
#include "MRCudaBasic.h"
#include "MRCudaBasic.hpp"
#include "MRCudaMath.cuh"
#include "MRCudaPolyline.h"

#include "MRMesh/MRAABBTreePolyline.h"
#include "MRMesh/MREndMill.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRTimer.h"

namespace MR::Cuda
{

struct ComputeToolDistance::Impl
{
    Polyline3DataHolder toolpath;
    Polyline2DataHolder toolProfile;
    DynamicArrayF output;

    Box2f toolBox;
    std::optional<std::variant<FlatEndMillTool, BallEndMillTool, BullNoseEndMillTool, ChamferEndMillTool>> toolSpec;

    void reset()
    {
        toolpath.reset();
        toolProfile.reset();
        output.resize( 0 );
        toolSpec.reset();
    }
};

ComputeToolDistance::ComputeToolDistance() : impl_( std::make_unique<Impl>() ) {}

ComputeToolDistance::~ComputeToolDistance() = default;

Expected<Vector3i> ComputeToolDistance::prepare( const Vector3i& dims, const Polyline3& toolpath,
    const EndMillTool& toolSpec )
{
    CUDA_LOGE_RETURN_UNEXPECTED( cudaSetDevice( 0 ) );

    impl_->reset();

    auto cudaToolpath = Polyline3DataHolder::fromLines( toolpath );
    MR_RETURN_IF_UNEXPECTED( cudaToolpath )
    impl_->toolpath = std::move( *cudaToolpath );

    using Cutter = EndMillCutter::Type;
    switch ( toolSpec.cutter.type )
    {
    case Cutter::Flat:
        impl_->toolSpec = FlatEndMillTool {
            .length = toolSpec.length,
            .radius = toolSpec.diameter / 2.f,
        };
        break;
    case Cutter::Ball:
        impl_->toolSpec = BallEndMillTool {
            .length = toolSpec.length,
            .radius = toolSpec.diameter / 2.f,
        };
        break;
    case Cutter::BullNose:
        impl_->toolSpec = BullNoseEndMillTool {
            .length = toolSpec.length,
            .radius = toolSpec.diameter / 2.f,
            .cornerRadius = toolSpec.cutter.cornerRadius,
        };
        break;
    case Cutter::Chamfer:
        impl_->toolSpec = ChamferEndMillTool {
            .length = toolSpec.length,
            .radius = toolSpec.diameter / 2.f,
            .endRadius = toolSpec.cutter.endDiameter / 2.f,
            .cutterHeight = toolSpec.getMinimalCutLength(),
        };
        break;
    case Cutter::Count:
        MR_UNREACHABLE
    }

    const auto layerSize = (size_t)dims.x * dims.y;
    const auto bufferSize = maxBufferSizeAlignedByBlock( getCudaSafeMemoryLimit(), dims, sizeof( float ) );
    CUDA_LOGE_RETURN_UNEXPECTED( impl_->output.resize( bufferSize ) );

    return Vector3i { dims.x, dims.y, int( bufferSize / layerSize ) };
}

Expected<Vector3i> ComputeToolDistance::prepare( const Vector3i& dims, const Polyline3& toolpath,
    const Polyline2& toolProfile )
{
    CUDA_LOGE_RETURN_UNEXPECTED( cudaSetDevice( 0 ) );

    impl_->reset();

    auto cudaToolpath = Polyline3DataHolder::fromLines( toolpath );
    MR_RETURN_IF_UNEXPECTED( cudaToolpath )
    impl_->toolpath = std::move( *cudaToolpath );

    auto cudaToolProfile = Polyline2DataHolder::fromLines( toolProfile );
    MR_RETURN_IF_UNEXPECTED( cudaToolProfile )
    impl_->toolProfile = std::move( *cudaToolProfile );

    const auto layerSize = (size_t)dims.x * dims.y;
    const auto bufferSize = maxBufferSizeAlignedByBlock( getCudaSafeMemoryLimit(), dims, sizeof( float ) );
    CUDA_LOGE_RETURN_UNEXPECTED( impl_->output.resize( bufferSize ) );

    return Vector3i { dims.x, dims.y, int( bufferSize / layerSize ) };
}

Expected<void> ComputeToolDistance::computeToolDistance( std::span<float> output, const Vector3i& dims, float voxelSize,
    const Vector3f& origin, float padding ) const
{
    MR_TIMER;

    CUDA_LOGE_RETURN_UNEXPECTED( cudaSetDevice( 0 ) );

    VolumeIndexer indexer( { dims.x, dims.y, dims.z }, voxelSize, { origin.x, origin.y, origin.z } );

    if ( impl_->toolSpec )
    {
        std::visit( overloaded { [&] ( auto&& tool )
        {
            computeToolDistanceKernel(
                impl_->output.data(), output.size(),
                indexer,
                impl_->toolpath,
                tool,
                padding
            );
        } }, *impl_->toolSpec );
    }
    else
    {
        const auto toolRadius = impl_->toolBox.max.x + padding;
        const auto toolMinHeight = impl_->toolBox.min.y - padding;
        const auto toolMaxHeight = impl_->toolBox.max.y + padding;

        computeToolDistanceKernel(
            impl_->output.data(), output.size(),
            indexer,
            impl_->toolpath,
            impl_->toolProfile,
            toolRadius, toolMinHeight, toolMaxHeight,
            padding
        );
    }
    CUDA_LOGE_RETURN_UNEXPECTED( cudaDeviceSynchronize() );
    CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );

    CUDA_LOGE_RETURN_UNEXPECTED( impl_->output.copyTo( output.data(), output.size() ) );

    return {};
}

} // namespace MR::Cuda

#endif
