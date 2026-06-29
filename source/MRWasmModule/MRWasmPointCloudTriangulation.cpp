#include "MRWasmBindings.h"

#include "MRMesh/MRPointCloudTriangulation.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <memory>
#include <optional>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_point_cloud_triangulation )
{
    emscripten::class_<TriangulationParameters>( "TriangulationParameters" )
        .constructor<>()
        .property( "numNeighbours", &TriangulationParameters::numNeighbours )
        .property( "radius", &TriangulationParameters::radius )
        .property( "critAngle", &TriangulationParameters::critAngle )
        .property( "critHoleLength", &TriangulationParameters::critHoleLength )
        .property( "automaticRadiusIncrease", &TriangulationParameters::automaticRadiusIncrease );

    emscripten::function( "triangulatePointCloud", +[]( const PointCloud& pc, const TriangulationParameters& params ) -> std::shared_ptr<Mesh>
    {
        auto res = triangulatePointCloud( pc, params );
        if ( !res )
            return nullptr;
        return std::make_shared<Mesh>( std::move( *res ) );
    } );
}
