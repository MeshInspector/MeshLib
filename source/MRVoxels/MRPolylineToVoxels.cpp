#include "MRPolylineToVoxels.h"
#include "MRFloatGrid.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshFillHole.h"
#include "MRVDBConversions.h"
#include "MRVoxelsVolume.h"
#include "MRMeshToDistanceVolume.h"

namespace MR
{

Mesh polylineToDegenerateMesh( const Polyline3& polyline )
{
    Mesh mesh;
    auto contours = polyline.topology.convertToContours<Vector3f>(
        [&points = polyline.points] ( VertId v )
    {
        return points[v];
    } );

    std::vector<EdgeId> newHoles;
    newHoles.reserve( contours.size() );
    for ( auto& cont : contours )
    {
        if ( cont[0] != cont.back() )
            cont.insert( cont.end(), cont.rbegin(), cont.rend() );
        newHoles.push_back( mesh.addSeparateEdgeLoop( cont ) );
    }

    for ( auto h : newHoles )
        makeDegenerateBandAroundHole( mesh, h );

    return mesh;
}

Expected<FloatGrid> polylineToDistanceField( const Polyline3& polyline, const PolylineToDistanceVolumeParams& params )
{
    MR_TIMER;

    const Mesh mesh = polylineToDegenerateMesh( polyline );
    return meshToDistanceField( mesh, {}, params.voxelSize, params.offsetCount, params.cb );
}

Expected<VdbVolume> polylineToVdbVolume( const Polyline3& polyline, const PolylineToDistanceVolumeParams& params )
{
    const Mesh mesh = polylineToDegenerateMesh( polyline );
    MeshToVolumeParams meshParams;
    meshParams.voxelSize = params.voxelSize;
    meshParams.surfaceOffset = params.offsetCount;
    meshParams.cb = params.cb;
    return meshToDistanceVdbVolume( mesh, meshParams );
}

Expected<SimpleVolume> polylineToSimpleVolume( const Polyline3& polyline, const PolylineToDistanceVolumeParams& params )
{
    PolylineToDistanceVolumeParams newParams = params;
    newParams.cb = subprogress( params.cb, 0.f, 0.5f );
    auto vdbVolume = polylineToVdbVolume( polyline, newParams );
    if ( !vdbVolume.has_value() )
        return unexpected( vdbVolume.error() );
    return vdbVolumeToSimpleVolume( *vdbVolume, {}, subprogress( params.cb, 0.5f, 1.f ) );
}

Expected<FunctionVolume> polylineToFunctionVolume( const Polyline3& polyline, const PolylineToFunctionVolumeParams& params )
{
    const Mesh mesh = polylineToDegenerateMesh( polyline );
    if ( !reportProgress( params.vol.cb, 0.5f ) )
        return unexpectedOperationCanceled();
    MeshToDistanceVolumeParams meshParams{ params.vol, params.dist, {} };
    return meshToDistanceFunctionVolume( mesh, meshParams );
}

}
