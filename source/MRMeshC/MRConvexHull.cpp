#include "MRConvexHull.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRConvexHull.h"
#include "MRMesh/MRMesh.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( PointCloud )

MRMesh* mrMakeConvexHullFromMesh( const MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW( makeConvexHull( mesh ) );
}

MRMesh* mrMakeConvexHullFromPointCloud( const MRPointCloud* pointCloud_ )
{
    ARG( pointCloud );
    RETURN_NEW( makeConvexHull( pointCloud ) );
}
