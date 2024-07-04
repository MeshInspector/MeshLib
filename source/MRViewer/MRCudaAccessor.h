#pragma once

#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRExpected.h"
#include <functional>
#include <memory>

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

    using CudaPointsToDistanceVolumeCallback = std::function<Expected<SimpleVolume>( const PointCloud& cloud, const PointsToDistanceVolumeParams& params )>;

    // setup functions
    MRVIEWER_API static void setCudaAvailable( bool val );
    MRVIEWER_API static void setCudaFreeMemoryFunc( CudaFreeMemoryFunc freeMemFunc );
    MRVIEWER_API static void setCudaFastWindingNumberConstructor( CudaFwnConstructor fwnCtor );
    MRVIEWER_API static void setCudaMeshProjectorConstructor( CudaMeshProjectorConstructor mpCtor );
    MRVIEWER_API static void setCudaPointsToDistanceVolumeCallback( CudaPointsToDistanceVolumeCallback callback );

    // Returns true if CUDA is available on this computer
    [[nodiscard]] MRVIEWER_API static bool isCudaAvailable();

    // Returns number of free bytes on cuda
    [[nodiscard]] MRVIEWER_API static size_t getCudaFreeMemory();

    // Returns cuda implementation of IFastWindingNumber
    [[nodiscard]] MRVIEWER_API static std::unique_ptr<IFastWindingNumber> getCudaFastWindingNumber( const Mesh& mesh );

    // Returns cuda implementation of IPointsToMeshProjector
    [[nodiscard]] MRVIEWER_API static std::unique_ptr<IPointsToMeshProjector> getCudaPointsToMeshProjector();

    // Returns cuda implementation of PointsToDistanceVolumeCallback
    [[nodiscard]] MRVIEWER_API static CudaPointsToDistanceVolumeCallback getCudaPointsToDistanceVolumeCallback();

    /// returns amount of required GPU memory for CudaFastWindingNumber internal data,
    /// note that this function will build mesh AABB Tree if it has not been built already
    /// \param mesh input mesh
    [[nodiscard]] MRVIEWER_API static size_t fastWindingNumberMeshMemory( const Mesh& mesh );

    /// returns amount of required GPU memory for CudaFastWindingNumber::calcFromGrid and CudaFastWindingNumber::calcFromGridWithDistances operations,
    /// note that this function will build mesh AABB Tree if it has not been built already
    /// \param dims dimensions of the grid
    [[nodiscard]] MRVIEWER_API static size_t fromGridMemory( const Mesh& mesh, const Vector3i& dims );

    /// <summary>
    /// returns amount of required GPU memory for CudaFastWindingNumber::calcFromVector operation
    /// note that this function will build mesh AABB Tree if it has not been built already
    /// </summary>
    /// <param name="inputSize">size of input vector</param>
    [[nodiscard]] MRVIEWER_API static size_t fromVectorMemory( const Mesh& mesh, size_t inputSize );

    /// <summary>
    /// returns amount of required GPU memory for CudaFastWindingNumber::calcSelfIntersections operation
    /// note that this function will build mesh AABB Tree if it has not been built already
    /// </summary>
    /// <param name="mesh">input mesh</param>
    [[nodiscard]] MRVIEWER_API static size_t selfIntersectionsMemory( const Mesh& mesh );
private:
    CudaAccessor() = default;
    ~CudaAccessor() = default;

    static CudaAccessor& instance_();

    bool isCudaAvailable_ = false;
    CudaFreeMemoryFunc freeMemFunc_;
    CudaFwnConstructor fwnCtor_;
    CudaMeshProjectorConstructor mpCtor_;
    CudaPointsToDistanceVolumeCallback pointsToDistanceVolumeCallback_;
};

} //namespace MR
