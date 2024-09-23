#include "MRPointsToMeshFusion.h"
#include "MRCalcDims.h"
#include "MRMarchingCubes.h"
#include "MRPointsToDistanceVolume.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRLocalTriangulations.h"
#include "MRMesh/MRPointCloudTriangulationHelpers.h"
#include "MRMesh/MRPointCloudMakeNormals.h"

namespace MR
{

Expected<Mesh> pointsToMeshFusion( const PointCloud & cloud, const PointsToMeshParameters& params )
{
    MR_TIMER

    PointsToDistanceVolumeParams p2vParams;

    VertNormals normals;
    if ( !cloud.hasNormals() )
    {
        auto optTriang = TriangulationHelpers::buildUnitedLocalTriangulations( cloud,
            { .radius = params.sigma }, subprogress( params.progress, 0.0f, 0.2f ) );

        if ( !optTriang )
            return unexpectedOperationCanceled();

        auto norms = makeOrientedNormals( cloud, *optTriang, subprogress( params.progress, 0.2f, 0.4f ) );
        if ( !norms )
            return unexpectedOperationCanceled();
        normals = std::move( *norms );
        p2vParams.ptNormals = &normals;
    }

    p2vParams.cb = p2vParams.ptNormals ? subprogress( params.progress, 0.4f, 0.65f ) : subprogress( params.progress, 0.0f, 0.5f );
    const auto box = cloud.getBoundingBox();
    const auto [origin, dimensions] = calcOriginAndDimensions( box, params.voxelSize );
    p2vParams.origin = origin;
    p2vParams.voxelSize = Vector3f::diagonal( params.voxelSize );
    p2vParams.dimensions = dimensions;
    p2vParams.sigma = params.sigma;
    p2vParams.minWeight = params.minWeight;

    MarchingCubesParams vmParams;
    vmParams.origin = p2vParams.origin;
    vmParams.iso = 0;
    vmParams.cb = subprogress( params.progress, p2vParams.ptNormals ? 0.65f : 0.5f, ( params.ptColors && params.vColors ) ? 0.9f : 1.0f );
    vmParams.lessInside = true;

    Expected<Mesh> res;
    if ( params.createVolumeCallback )
    {
        res = params.createVolumeCallback( cloud, p2vParams ).and_then( [&vmParams] ( SimpleVolumeMinMax&& volume )
        {
            vmParams.freeVolume = [&volume]
            {
                Timer t( "~SimpleVolume" );
                volume = {};
            };
            return marchingCubes( volume, vmParams );
        } );
    }
    else
        res = marchingCubes( pointsToDistanceFunctionVolume( cloud, p2vParams ), vmParams );

    if ( res && params.ptColors && params.vColors )
    {
        auto optColors = calcAvgColors( cloud, *params.ptColors, res->points, res->topology.getValidVerts(),
            params.sigma, subprogress( params.progress, 0.9f, 1.0f ) );
        if ( optColors )
            *params.vColors = std::move( optColors.value() );
        else
            res = unexpected( std::move( optColors.error() ) );
    }

    return res;
}

} //namespace MR
