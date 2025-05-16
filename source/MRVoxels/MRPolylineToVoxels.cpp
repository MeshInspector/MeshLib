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

Expected<SimpleVolume> polylineToSimpleVolume( const Polyline3& polyline, const PolylineToVolumeParams& params )
{
    const Mesh mesh = polylineToDegenerateMesh( polyline );
    MeshToDistanceVolumeParams meshParams;
    meshParams.vol = params.vol;
    meshParams.dist = { params.dist };
    return meshToDistanceVolume( mesh, meshParams );
}

Expected<FunctionVolume> polylineToFunctionVolume( const Polyline3& polyline, const PolylineToVolumeParams& params )
{
    const Mesh mesh = polylineToDegenerateMesh( polyline );
    MeshToDistanceVolumeParams meshParams;
    meshParams.vol = params.vol;
    meshParams.dist = { params.dist };
    return meshToDistanceFunctionVolume( mesh, meshParams );
}

}
