#pragma once

#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include <functional>
#include <memory>

namespace MR
{

class IPointsToMeshProjector;

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

    // setup functions
    MRVIEWER_API static void setCudaAvailable( bool val );
    MRVIEWER_API static void setCudaFreeMemoryFunc( CudaFreeMemoryFunc freeMemFunc );
    MRVIEWER_API static void setCudaFastWindingNumberConstructor( CudaFwnConstructor fwnCtor );
    MRVIEWER_API static void setCudaMeshProjectorConstructor( CudaMeshProjectorConstructor mpCtor );

    // Returns true if CUDA is available on this computer
    MRVIEWER_API static bool isCudaAvailable();
    // Returns number of free bytes on cuda
    MRVIEWER_API static size_t getCudaFreeMemory();
    // Returns cuda implementation of IFastWindingNumber
    MRVIEWER_API static std::unique_ptr<IFastWindingNumber> getCudaFastWindingNumber( const Mesh& mesh );
    // Returns cuda implementation of IPointsToMeshProjector
    MRVIEWER_API static std::unique_ptr<IPointsToMeshProjector> getCudaPointsToMeshProjector();

private:
    CudaAccessor() = default;
    ~CudaAccessor() = default;

    static CudaAccessor& instance_();

    bool isCudaAvailable_ = false;
    CudaFreeMemoryFunc freeMemFunc_;
    CudaFwnConstructor fwnCtor_;
    CudaMeshProjectorConstructor mpCtor_;
};

} //namespace MR
