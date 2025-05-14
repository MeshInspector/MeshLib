#include "MRPointsToVoxels.h"
#include "MRFloatGrid.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshFillHole.h"
#include "MRVDBConversions.h"

namespace MR
{

Expected<FloatGrid> polylineToDistanceField( const Polyline3& polyline, const Vector3f& voxelSize, float offsetCount /*= 3*/, ProgressCallback cb /*= {}*/ )
{
    MR_TIMER;

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

    return meshToDistanceField( mesh, {}, voxelSize, offsetCount, cb );
}

}
