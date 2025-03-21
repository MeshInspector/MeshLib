#include "MRSurfaceLineOffset.h"
#include "MRContoursCut.h"

namespace MR
{

Expected<Contours3f> offsetSurfaceLine( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine, const std::function<float( int )>& offsetAtPoint )
{
    auto offsetRes = convertMeshTriPointsSurfaceOffsetToMeshContours( mesh, surfaceLine, offsetAtPoint );
    if ( !offsetRes.has_value() )
        return unexpected( std::move( offsetRes.error() ) );

    return extractMeshContours( std::move( *offsetRes ) );
}

Expected<Contours3f> offsetSurfaceLine( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine, float offset )
{
    return offsetSurfaceLine( mesh, surfaceLine, [offset] ( int )->float
    {
        return offset;
    } );
}

}