#include "MRWasmBindings.h"

#include "MRMesh/MRPointCloudRadius.h"
#include "MRMesh/MRPointCloud.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_point_cloud_radius )
{
    emscripten::function( "findAvgPointsRadius", +[]( const PointCloud& pointCloud, int avgPoints )
    {
        return findAvgPointsRadius( pointCloud, avgPoints );
    } );
    emscripten::function( "findAvgPointsRadius", +[]( const PointCloud& pointCloud, int avgPoints, int samples )
    {
        return findAvgPointsRadius( pointCloud, avgPoints, samples );
    } );
}
