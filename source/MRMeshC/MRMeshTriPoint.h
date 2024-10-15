#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRTriPoint.h"

MR_EXTERN_C_BEGIN

/// encodes a point inside a triangular mesh face using barycentric coordinates
/// \details Notations used below: \n
///   v0 - the value in org( e ) \n
///   v1 - the value in dest( e ) \n
///   v2 - the value in dest( next( e ) )
typedef struct MRMeshTriPoint
{
    /// left face of this edge is considered
    MREdgeId e;
    /// barycentric coordinates
    /// \details a in [0,1], a=0 => point is on next( e ) edge, a=1 => point is in dest( e )
    /// b in [0,1], b=0 => point is on e edge, b=1 => point is in dest( next( e ) )
    /// a+b in [0,1], a+b=0 => point is in org( e ), a+b=1 => point is on prev( e.sym() ) edge
    MRTriPointf bary;
} MRMeshTriPoint;

MR_EXTERN_C_END
