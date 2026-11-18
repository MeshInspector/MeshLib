#include "MRWasmBindings.h"

#include "MRMesh/MRConvexHull.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPointCloud.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_convex_hull )
{
    emscripten::function( "makeConvexHullFromMesh", +[]( std::shared_ptr<Mesh> mesh )
    {
        return std::make_shared<Mesh>( makeConvexHull( *mesh ) );
    } );
    emscripten::function( "makeConvexHullFromPoints", +[]( const PointCloud& pointCloud )
    {
        return std::make_shared<Mesh>( makeConvexHull( pointCloud ) );
    } );
}
