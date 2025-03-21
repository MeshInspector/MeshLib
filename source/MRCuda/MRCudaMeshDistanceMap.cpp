#include "MRCudaMeshDistanceMap.h"
#include "MRCudaMeshDistanceMap.cuh"

#include "MRCudaBasic.cuh"
#include "MRCudaBasic.h"

#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRIntersectionPrecomputes.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRVector3.h"

namespace MR::Cuda
{

Expected<DistanceMap> computeDistanceMap( const MR::Mesh& mesh, const MR::MeshToDistanceMapParams& params, ProgressCallback cb /*= {}*/, std::vector<MR::MeshTriPoint>* outSamples /*= nullptr */ )
{
    MR_TIMER;

    if ( params.resolution.x <= 0 || params.resolution.y <= 0 )
        return {};

    DistanceMap distMap( params.resolution.x, params.resolution.y );

    // precomputed some values
    MR::IntersectionPrecomputes<double> prec( Vector3d( params.direction ) );

    auto ori = params.orgPoint;
    float shift = 0.f;
    if ( params.allowNegativeValues )
    {
        AffineXf3f xf( Matrix3f( params.xRange.normalized(), params.yRange.normalized(), params.direction.normalized() ), Vector3f() );
        Box3f box = mesh.computeBoundingBox( &xf );
        shift = dot( params.direction, ori - box.min );
        if ( shift > 0.f )
        {
            ori -= params.direction * shift;
        }
        else
        {
            shift = 0.0f;
        }
    }

    if ( !reportProgress( cb, 0.1f ) )
        return unexpectedOperationCanceled();

    const AABBTree& tree = mesh.getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& meshPoints = mesh.points;
    const auto tris = mesh.topology.getTriangulation();

    DynamicArray<float3> cudaMeshPoints;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaMeshPoints.fromVector( meshPoints.vec_ ) );

    DynamicArray<Node3> cudaNodes;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaNodes.fromVector( nodes.vec_ ) );

    DynamicArray<FaceToThreeVerts> cudaFaces;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaFaces.fromVector( tris.vec_ ) );

    MeshToDistanceMapParams cudaParams {
        .xRange = { params.xRange.x, params.xRange.y, params.xRange.z },
        .yRange = { params.yRange.x, params.yRange.y, params.yRange.z },
        .direction = { params.direction.x, params.direction.y, params.direction.z },
        .orgPoint = { ori.x, ori.y, ori.z },
        .resolution = { params.resolution.x, params.resolution.y },
        .minValue = params.minValue,
        .maxValue = params.maxValue,
        .useDistanceLimits = params.useDistanceLimits,
        .allowNegativeValues = params.allowNegativeValues,
    };

    const auto totalSize = distMap.size();
    const auto bufferSize = maxBufferSizeAlignedByBlock( getCudaSafeMemoryLimit(), distMap.dims(), sizeof( float ) + ( outSamples ? sizeof( Cuda::MeshTriPoint ) : 0 ) );

    DynamicArray<float> result;
    CUDA_LOGE_RETURN_UNEXPECTED( result.resize( bufferSize ) );
    std::vector<float> vec( totalSize );

    DynamicArray<Cuda::MeshTriPoint> outTriPoints;
    if ( outSamples )
    {
        CUDA_LOGE_RETURN_UNEXPECTED( outTriPoints.resize( bufferSize ) );
        outSamples->resize( totalSize );
    }

    if ( !reportProgress( cb, 0.4f ) )
        return unexpectedOperationCanceled();

    IntersectionPrecomputes cudaPrec {
        .dir = { params.direction.x, params.direction.y, params.direction.z },
        .invDir = { float( prec.invDir.x ), float( prec.invDir.y ), float( prec.invDir.z ) },
        .maxDimIdxZ = prec.maxDimIdxZ,
        .idxX = prec.idxX,
        .idxY = prec.idxY,
        .sign = { prec.sign.x, prec.sign.y, prec.sign.z },
        .Sx = float( prec.Sx ),
        .Sy = float( prec.Sy ),
        .Sz = float( prec.Sz ),
    };

    const auto cb1 = subprogress( cb, 0.40f, 1.00f );
    const auto iterCount = chunkCount( totalSize, bufferSize );
    size_t iterIndex = 0;

    for ( const auto [offset, size] : splitByChunks( totalSize, bufferSize ) )
    {
        const auto cb2 = subprogress( cb1, iterIndex++, iterCount );

        computeMeshDistanceMapKernel( cudaNodes.data(), cudaMeshPoints.data(), cudaFaces.data(), cudaParams, cudaPrec, shift, result.data(), outSamples ? outTriPoints.data() : nullptr, size, offset );
        CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );
        if ( !reportProgress( cb2, 0.33f ) )
            return unexpectedOperationCanceled();

        CUDA_LOGE_RETURN_UNEXPECTED( result.copyTo( vec.data() + offset, size ) );
        if ( outSamples )
            CUDA_LOGE_RETURN_UNEXPECTED( outTriPoints.copyTo( outSamples->data() + offset, size ) );
        if ( !reportProgress( cb2, 1.00f ) )
            return unexpectedOperationCanceled();
    }

    distMap.set( std::move( vec ) );
    return distMap;
}

size_t computeDistanceMapHeapBytes( const MR::Mesh& mesh, const MR::MeshToDistanceMapParams& params, bool needOutSamples /*= false */ )
{
    const AABBTree& tree = mesh.getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& meshPoints = mesh.points;
    constexpr size_t cMinRowCount = 10;
    size_t size = cMinRowCount * params.resolution.y;
    return
        nodes.size() * sizeof( Cuda::Node3 ) +
        meshPoints.size() * sizeof( float3 ) +
        mesh.topology.numValidFaces() * sizeof( FaceToThreeVerts ) +
        size * sizeof( float ) +
        ( needOutSamples ? ( size * sizeof( Cuda::MeshTriPoint ) ) : 0 );
}

} // namespace MR::Cuda
