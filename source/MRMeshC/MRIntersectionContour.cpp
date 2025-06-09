#include "MRIntersectionContour.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRIntersectionContour.h"

using namespace MR;

REGISTER_AUTO_CAST( ContinuousContours )
REGISTER_AUTO_CAST( MeshTopology )
REGISTER_VECTOR_LIKE( MRVectorVarEdgeTri, VarEdgeTri )

MRContinuousContour mrContinuousContoursGet( const MRContinuousContours* contours_, size_t index )
{
    ARG( contours );
    RETURN_VECTOR( contours[index] );
}

size_t mrContinuousContoursSize( const MRContinuousContours* contours_ )
{
    ARG( contours );
    return contours.size();
}

void mrContinuousContoursFree( MRContinuousContours* contours_ )
{
    ARG_PTR( contours );
    delete contours;
}

MRContinuousContours* mrOrderIntersectionContours( const MRMeshTopology* topologyA_, const MRMeshTopology* topologyB_, const MRPreciseCollisionResult* intersections_ )
{
    ARG( topologyA ); ARG( topologyB ); ARG( intersections );
    RETURN_NEW( orderIntersectionContours( topologyA, topologyB, intersections ) );
}
