#include "MRPointsToMeshFusion.h"
#include "MRPointCloud.h"
#include "MRMesh.h"
#include "MRMarchingCubes.h"
#include "MRBox.h"
#include "MRColor.h"
#include "MRTimer.h"

namespace MR
{

Expected<Mesh> pointsToMeshFusion( const PointCloud & cloud, const PointsToMeshParameters& params )
{
    MR_TIMER

    PointsToDistanceVolumeParams p2vParams;
    p2vParams.cb = subprogress( params.progress, 0.0f, 0.5f );
    auto box = cloud.getBoundingBox();
    auto expansion = Vector3f::diagonal( 2 * params.voxelSize );
    p2vParams.origin = box.min - expansion;
    p2vParams.voxelSize = Vector3f::diagonal( params.voxelSize );
    p2vParams.dimensions = Vector3i( ( box.max + expansion - p2vParams.origin ) / params.voxelSize ) + Vector3i::diagonal( 1 );
    p2vParams.sigma = params.sigma;
    p2vParams.minWeight = params.minWeight;

    MarchingCubesParams vmParams;
    vmParams.origin = p2vParams.origin;
    vmParams.iso = 0;
    vmParams.cb = subprogress( params.progress, 0.5f, ( params.ptColors && params.vColors ) ? 0.9f : 1.0f );
    vmParams.lessInside = true;

    auto res = ( params.createVolumeCallback ?
        params.createVolumeCallback( cloud, p2vParams ) : pointsToDistanceVolume( cloud, p2vParams ) ).and_then( [vmParams] ( auto&& volume ) { return marchingCubes( volume, vmParams ); } );

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
