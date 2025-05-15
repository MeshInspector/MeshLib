#include "MRPointsToVoxels.h"
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

Expected<FloatGrid> polylineToDistanceField( const Polyline3& polyline, const Vector3f& voxelSize, float offsetCount /*= 3*/, ProgressCallback cb /*= {}*/ )
{
    MR_TIMER;

    const Mesh mesh = polylineToDegenerateMesh( polyline );
    return meshToDistanceField( mesh, {}, voxelSize, offsetCount, cb );
}

Expected<VdbVolume> polylineToVdbVolume( const Polyline3& polyline, const Vector3f& voxelSize, float offsetCount /*= 3*/, ProgressCallback cb /*= {} */ )
{
    const Mesh mesh = polylineToDegenerateMesh( polyline );
    MeshToVolumeParams params;
    params.voxelSize = voxelSize;
    params.surfaceOffset = offsetCount;
    params.cb = cb;
    return meshToDistanceVdbVolume( mesh, params );
}

Expected<SimpleVolume> polylineToSimpleVolume( const Polyline3& polyline, const Vector3f& voxelSize, float offsetCount /*= 3*/, ProgressCallback cb /*= {} */ )
{
    auto vdbVolume = polylineToVdbVolume( polyline, voxelSize, offsetCount, subprogress( cb, 0.f, 0.9f ) );
    if ( !vdbVolume.has_value() )
        return unexpected( vdbVolume.error() );
    return vdbVolumeToSimpleVolume( *vdbVolume, {}, subprogress( cb, 0.9f, 1.f ) );
}

Expected<FunctionVolume> polylineToFunctionVolume( const Polyline3& polyline, const Vector3f& voxelSize, ProgressCallback cb /*= {} */ )
{
    const Mesh mesh = polylineToDegenerateMesh( polyline );
    MeshToDistanceVolumeParams params;
    params.vol.voxelSize = voxelSize;
    params.vol.cb = cb;
    return meshToDistanceFunctionVolume( mesh, params );
}

}
