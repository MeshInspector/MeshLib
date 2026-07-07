#include "MRWasmBindings.h"

#include "MRMesh/MROneMeshContours.h"
#include "MRMesh/MRIntersectionContour.h"
#include "MRMesh/MRPrecisePredicates3.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <utility>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_one_mesh_contours )
{
    // Opaque carrier: std::vector<OneMeshContour>; produced here, consumed by cutMesh.
    emscripten::class_<OneMeshContours>( "OneMeshContours" );

    emscripten::function( "getOneMeshIntersectionContours", +[]( const Mesh& meshA, const Mesh& meshB,
        const ContinuousContours& contours, bool getMeshAIntersections, const CoordinateConverters& converters )
    {
        OneMeshContours outA, outB;
        getOneMeshIntersectionContours( meshA, meshB, contours,
            getMeshAIntersections ? &outA : nullptr,
            getMeshAIntersections ? nullptr : &outB, converters );
        return getMeshAIntersections ? std::move( outA ) : std::move( outB );
    } );
}
