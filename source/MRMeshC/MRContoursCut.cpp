#include "MRContoursCut.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRContoursCut.h"

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( ContinuousContours )
REGISTER_AUTO_CAST( ConvertToFloatVector )
REGISTER_AUTO_CAST( ConvertToIntVector )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( OneMeshContours )

static_assert( sizeof( MROneMeshIntersection ) == sizeof( OneMeshIntersection ) );

MROneMeshContour mrOneMeshContoursGet( const MROneMeshContours* contours_, size_t index )
{
    ARG( contours );
    const auto& result = contours[index];
    return {
        .intersections = cast_to<MRVectorOneMeshIntersection>( vector_ref_wrapper( result.intersections ) ),
        .closed = result.closed,
    };
}

size_t mrOneMeshContoursSize( const MROneMeshContours* contours_ )
{
    ARG( contours );
    return contours.size();
}

void mrOneMeshContoursFree( MROneMeshContours* contours_ )
{
    ARG_PTR( contours );
    delete contours;
}

MROneMeshContours* mrGetOneMeshIntersectionContours( const MRMesh* meshA_, const MRMesh* meshB_, const MRContinuousContours* contours_, bool getMeshAIntersections, const MRCoordinateConverters* converters_, const MRAffineXf3f* rigidB2A_ )
{
    ARG( meshA ); ARG( meshB ); ARG( contours ); ARG_PTR( rigidB2A );
    const CoordinateConverters converters {
        .toInt = *auto_cast( converters_->toInt ),
        .toFloat = *auto_cast( converters_->toFloat ),
    };
    OneMeshContours res;
    getOneMeshIntersectionContours( meshA, meshB, contours,
        getMeshAIntersections ? &res : nullptr,
        getMeshAIntersections ? nullptr : &res,
        converters, rigidB2A );
    RETURN_NEW( res );
}
