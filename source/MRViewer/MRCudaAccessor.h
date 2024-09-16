#pragma once

#include "MRViewerFwd.h"
#include "MRMesh/MRExpected.h"

#include <functional>
#include <memory>

#ifndef MRVIEWER_NO_VOXELS
#include "MRVoxels/MRVoxelsFwd.h"
#endif

namespace MR
{

struct PointsToDistanceVolumeParams;
/// The purpose of this class is to access CUDA algorithms without explicit dependency on MRCuda
class MRVIEWER_CLASS CudaAccessor
{
public:
    /// Returns amount of free memory on GPU
    using CudaFreeMemoryFunc = std::function<size_t()>;
    /// Returns specific implementation of IFastWindingNumber interface that computes windings on GPU
    using CudaFwnConstructor = std::function<std::unique_ptr<IFastWindingNumber>( const Mesh& )>;
    /// Returns specific implementation of IPointsToMeshProjector interface projects on GPU
    using CudaMeshProjectorConstructor = std::function<std::unique_ptr<IPointsToMeshProjector>()>;

#ifndef MRVIEWER_NO_VOXELS
    using CudaPointsToDistanceVolumeCallback = std::function<Expected<SimpleVolumeMinMax>( const PointCloud& cloud, const PointsToDistanceVolumeParams& params )>;
#endif

    // setup functions
    MRVIEWER_API static void setCudaAvailable( bool val, int maxDriverVersion, int runtimeVersion, int computeMajor, int computeMinor );
    MRVIEWER_API static void setCudaFreeMemoryFunc( CudaFreeMemoryFunc freeMemFunc );
    MRVIEWER_API static void setCudaFastWindingNumberConstructor( CudaFwnConstructor fwnCtor );
    MRVIEWER_API static void setCudaMeshProjectorConstructor( CudaMeshProjectorConstructor mpCtor );

#ifndef MRVIEWER_NO_VOXELS
    MRVIEWER_API static void setCudaPointsToDistanceVolumeCallback( CudaPointsToDistanceVolumeCallback callback );
#endif

    // Returns true if CUDA is available on this computer
    [[nodiscard]] MRVIEWER_API static bool isCudaAvailable();

    // Returns maximum supported by driver version
    [[nodiscard]] MRVIEWER_API static int getCudaMaxDriverSupportedVersion();

    // Returns version of current runtime
    [[nodiscard]] MRVIEWER_API static int getCudaRuntimeVersion();

    // Returns Compute Capability major version of current device
    [[nodiscard]] MRVIEWER_API static int getComputeCapabilityMajor();

    // Returns Compute Capability minor version of current device
    [[nodiscard]] MRVIEWER_API static int getComputeCapabilityMinor();

    // Returns number of free bytes on cuda
    [[nodiscard]] MRVIEWER_API static size_t getCudaFreeMemory();

    // Returns cuda implementation of IFastWindingNumber
    [[nodiscard]] MRVIEWER_API static std::unique_ptr<IFastWindingNumber> getCudaFastWindingNumber( const Mesh& mesh );

    // Returns cuda implementation of IPointsToMeshProjector
    [[nodiscard]] MRVIEWER_API static std::unique_ptr<IPointsToMeshProjector> getCudaPointsToMeshProjector();

#ifndef MRVIEWER_NO_VOXELS
    // Returns cuda implementation of PointsToDistanceVolumeCallback
    [[nodiscard]] MRVIEWER_API static CudaPointsToDistanceVolumeCallback getCudaPointsToDistanceVolumeCallback();
#endif

    /// returns amount of required GPU memory for CudaFastWindingNumber internal data,
    /// \param mesh input mesh
    [[nodiscard]] MRVIEWER_API static size_t fastWindingNumberMeshMemory( const Mesh& mesh );

    /// returns amount of required GPU memory for CudaFastWindingNumber::calcFromGrid and CudaFastWindingNumber::calcFromGridWithDistances operations,
    /// \param dims dimensions of the grid
    [[nodiscard]] MRVIEWER_API static size_t fromGridMemory( const Mesh& mesh, const Vector3i& dims );

    /// <summary>
    /// returns amount of required GPU memory for CudaFastWindingNumber::calcFromVector operation
    /// </summary>
    /// <param name="inputSize">size of input vector</param>
    [[nodiscard]] MRVIEWER_API static size_t fromVectorMemory( const Mesh& mesh, size_t inputSize );

    /// <summary>
    /// returns amount of required GPU memory for CudaFastWindingNumber::calcSelfIntersections operation
    /// </summary>
    /// <param name="mesh">input mesh</param>
    [[nodiscard]] MRVIEWER_API static size_t selfIntersectionsMemory( const Mesh& mesh );
private:
    CudaAccessor() = default;
    ~CudaAccessor() = default;

    static CudaAccessor& instance_();

    bool isCudaAvailable_ = false;
    int maxDriverVersion_ = 0;
    int runtimeVersion_ = 0;
    int computeMajor_ = 0;
    int computeMinor_ = 0;
    CudaFreeMemoryFunc freeMemFunc_;
    CudaFwnConstructor fwnCtor_;
    CudaMeshProjectorConstructor mpCtor_;
#ifndef MRVIEWER_NO_VOXELS
    CudaPointsToDistanceVolumeCallback pointsToDistanceVolumeCallback_;
#endif
};

} //namespace MR
