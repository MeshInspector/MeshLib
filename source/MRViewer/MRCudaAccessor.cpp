#include "MRCudaAccessor.h"
#include "MRMesh/MRPointsToMeshProjector.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRAABBTreePoints.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRAABBTreeMaker.h"
#include "MRMesh/MRDipole.h"

#ifndef MRVIEWER_NO_VOXELS
#include "MRVoxels/MRVoxelsVolume.h"
#endif

namespace MR
{

void CudaAccessor::setCudaAvailable( bool val, int maxDriverVersion, int runtimeVersion, int computeMajor, int computeMinor )
{
    auto& inst = instance_();
    inst.isCudaAvailable_ = val;
    inst.maxDriverVersion_ = maxDriverVersion;
    inst.runtimeVersion_ = runtimeVersion;
    inst.computeMajor_ = computeMajor;
    inst.computeMinor_ = computeMinor;
}

void CudaAccessor::setCudaFreeMemoryFunc( CudaFreeMemoryFunc freeMemFunc )
{
    instance_().freeMemFunc_ = freeMemFunc;
}

void CudaAccessor::setCudaFastWindingNumberConstructor( CudaFwnConstructor fwnCtor )
{
    instance_().fwnCtor_ = fwnCtor;
}

void CudaAccessor::setCudaMeshProjectorConstructor( CudaMeshProjectorConstructor mpCtor )
{
    instance_().mpCtor_ = mpCtor;
}

#ifndef MRVIEWER_NO_VOXELS
void CudaAccessor::setCudaPointsToDistanceVolumeCallback( CudaPointsToDistanceVolumeCallback callback )
{
    instance_().pointsToDistanceVolumeCallback_ = callback;
}

void CudaAccessor::setCudaPointsToDistanceVolumeByPartsCallback( CudaPointsToDistanceVolumeByPartsCallback callback )
{
    instance_().pointsToDistanceVolumeByPartsCallback_ = callback;
}
#endif

bool CudaAccessor::isCudaAvailable()
{
    auto& inst = instance_();
    return inst.isCudaAvailable_;
}

int CudaAccessor::getCudaMaxDriverSupportedVersion()
{
    return instance_().maxDriverVersion_;
}

int CudaAccessor::getCudaRuntimeVersion()
{
    return instance_().runtimeVersion_;
}

int CudaAccessor::getComputeCapabilityMajor()
{
    return instance_().computeMajor_;
}

int CudaAccessor::getComputeCapabilityMinor()
{
    return instance_().computeMinor_;
}

size_t CudaAccessor::getCudaFreeMemory()
{
    auto& inst = instance_();
    if ( !inst.freeMemFunc_ )
        return 0;
    return inst.freeMemFunc_();
}

std::unique_ptr<IFastWindingNumber> CudaAccessor::getCudaFastWindingNumber( const Mesh& mesh )
{
    auto& inst = instance_();
    if ( !inst.fwnCtor_ )
        return {};
    return inst.fwnCtor_( mesh );
}

std::unique_ptr<IPointsToMeshProjector> CudaAccessor::getCudaPointsToMeshProjector()
{
    auto& inst = instance_();
    if ( !inst.mpCtor_ )
        return {};
    return inst.mpCtor_();
}

#ifndef MRVIEWER_NO_VOXELS
CudaAccessor::CudaPointsToDistanceVolumeCallback CudaAccessor::getCudaPointsToDistanceVolumeCallback()
{
    auto& inst = instance_();
    if ( !inst.pointsToDistanceVolumeCallback_ )
        return {};

    return inst.pointsToDistanceVolumeCallback_;
}

CudaAccessor::CudaPointsToDistanceVolumeByPartsCallback CudaAccessor::getCudaPointsToDistanceVolumeByPartsCallback()
{
    auto& inst = instance_();
    if ( !inst.pointsToDistanceVolumeByPartsCallback_ )
        return {};

    return inst.pointsToDistanceVolumeByPartsCallback_;
}
#endif

size_t CudaAccessor::fastWindingNumberMeshMemory( const Mesh& mesh )
{
    size_t treeNodesSize = getNumNodes( mesh.topology.numValidFaces() );
    size_t memoryAmount = treeNodesSize * sizeof( Dipole );
    memoryAmount += mesh.points.size() * sizeof( Vector3f );
    memoryAmount += treeNodesSize * sizeof( AABBTree::Node );
    memoryAmount += mesh.topology.faceSize() * sizeof( Vector3i );
    return memoryAmount;
}

size_t CudaAccessor::fromGridMemory( const Mesh& mesh, const Vector3i& dims )
{
    constexpr size_t cMinLayerCount = 10;
    return
        fastWindingNumberMeshMemory( mesh )
        + std::min( (size_t)dims.z, cMinLayerCount ) * dims.x * dims.y * sizeof( float );
}

size_t CudaAccessor::fromVectorMemory( const Mesh& mesh, size_t inputSize )
{
    constexpr size_t cMinCudaBufferSize = 1 << 24; // 16 MiB
    return
        fastWindingNumberMeshMemory( mesh )
        + std::min( inputSize * ( sizeof( float ) + sizeof( Vector3f ) ), cMinCudaBufferSize );
}

size_t CudaAccessor::selfIntersectionsMemory( const Mesh& mesh )
{
    constexpr size_t cMinCudaBufferSize = 1 << 24; // 16 MiB
    return
        fastWindingNumberMeshMemory( mesh )
        + std::min( mesh.topology.faceSize() * sizeof( float ), cMinCudaBufferSize );
}

size_t CudaAccessor::pointsToDistanceVolumeMemory( const PointCloud& pointCloud, const Vector3i& dims, const VertNormals* ptNormals )
{
    constexpr size_t cMinLayerCount = 10;

    const auto& tree = pointCloud.getAABBTree();
    const auto& nodes = tree.nodes();

    return
        nodes.size() * sizeof( AABBTreePoints::Node )
        + tree.orderedPoints().size() * sizeof( AABBTreePoints::Point )
        + ( ptNormals ? ptNormals->size() : pointCloud.normals.size() ) * sizeof( Vector3f )
        + std::min( (size_t)dims.z, cMinLayerCount ) * dims.x * dims.y * sizeof( float )
    ;
}

CudaAccessor& CudaAccessor::instance_()
{
    static CudaAccessor instance;
    return instance;
}

}