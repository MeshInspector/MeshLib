#include "MRCudaMeshDistanceMap.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRIntersectionPrecomputes.h"
#include "MRMesh/MRAABBTree.h"
#include "MRCudaMeshDistanceMap.cuh"
#include "MRCudaBasic.cuh"

namespace MR
{

namespace Cuda
{

DistanceMap computeDistanceMap( const MR::Mesh& mesh, const MR::MeshToDistanceMapParams& params, ProgressCallback cb /*= {}*/, std::vector<MR::MeshTriPoint>* outSamples /*= nullptr */ )
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
        return {};

    const AABBTree& tree = mesh.getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& meshPoints = mesh.points;
    const auto tris = mesh.topology.getTriangulation();

    DynamicArray<float3> cudaMeshPoints;
    DynamicArray<Node3> cudaNodes;
    DynamicArray<FaceToThreeVerts> cudaFaces;

    cudaMeshPoints.fromVector( meshPoints.vec_ );
    cudaNodes.fromVector( nodes.vec_ );
    cudaFaces.fromVector( tris.vec_ );

    MR::Cuda::MeshToDistanceMapParams cudaParams;
    cudaParams.allowNegativeValues = params.allowNegativeValues;
    cudaParams.maxValue = params.maxValue;
    cudaParams.minValue = params.minValue;
    cudaParams.useDistanceLimits = params.useDistanceLimits;
    cudaParams.resolution = { params.resolution.x,params.resolution.y };
    cudaParams.direction = { params.direction.x,params.direction.y,params.direction.z };
    cudaParams.orgPoint = { ori.x,ori.y,ori.z };
    cudaParams.xRange = { params.xRange.x,params.xRange.y,params.xRange.z };
    cudaParams.yRange = { params.yRange.x,params.yRange.y,params.yRange.z };

    DynamicArray<float> result;
    DynamicArray<MR::Cuda::MeshTriPoint> outTriPoints;
    result.resize( distMap.size() );
    if ( outSamples )
        outTriPoints.resize( distMap.size() );

    if ( !reportProgress( cb, 0.4f ) )
        return {};

    MR::Cuda::IntersectionPrecomputes cudaPrec;
    cudaPrec.dir = { params.direction.x,params.direction.y,params.direction.z };
    cudaPrec.idxX = prec.idxX;
    cudaPrec.idxY = prec.idxY;
    cudaPrec.invDir = { float( prec.invDir.x ),float( prec.invDir.y ),float( prec.invDir.z ) };
    cudaPrec.maxDimIdxZ = prec.maxDimIdxZ;
    cudaPrec.sign = { prec.sign.x,prec.sign.y,prec.sign.z };
    cudaPrec.Sx = float( prec.Sx );
    cudaPrec.Sy = float( prec.Sy );
    cudaPrec.Sz = float( prec.Sz );

    CUDA_EXEC( computeMeshDistanceMapKernel( cudaNodes.data(), cudaMeshPoints.data(), cudaFaces.data(), cudaParams, cudaPrec, shift, result.data(), outSamples ? outTriPoints.data() : nullptr ) );

    if ( !reportProgress( cb, 0.6f ) )
        return {};

    CUDA_EXEC( result.toBytes( ( uint8_t* )distMap.data() ) );
    if ( outSamples )
    {
        CUDA_EXEC( outTriPoints.toVector( *outSamples ) );
    }
    if ( !reportProgress( cb, 1.0f ) )
        return {};
    return distMap;
}

size_t computeDistanceMapHeapBytes( const MR::Mesh& mesh, const MR::MeshToDistanceMapParams& params, bool needOutSamples /*= false */ )
{
    const AABBTree& tree = mesh.getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& meshPoints = mesh.points;
    size_t size = size_t( params.resolution.x ) * params.resolution.y;
    return
        nodes.size() * sizeof( Cuda::Node3 ) +
        meshPoints.size() * sizeof( float3 ) +
        mesh.topology.numValidFaces() * sizeof( FaceToThreeVerts ) +
        size * sizeof( float ) +
        ( needOutSamples ? ( size * sizeof( Cuda::MeshTriPoint ) ) : 0 );
}

}

}