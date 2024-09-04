#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

/// barycentric coordinates:
/// a+b in [0,1], a+b=0 => point is in v0, a+b=1 => point is on [v1,v2] edge
public value struct TriPoint
{
    ///< a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
    float a;
    ///< b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2
    float b;
    /// given three values in three vertices, computes interpolated value at this barycentric coordinates
    Vector3f^ Interpolate( Vector3f^ p0, Vector3f^ p1, Vector3f^ p2 );
};

/// encodes a point inside a triangular mesh face using barycentric coordinates
/// \ingroup MeshGroup
/// \details Notations used below: \n
///   v0 - the value in org( e ) \n
///   v1 - the value in dest( e ) \n
///   v2 - the value in dest( next( e ) )
public ref struct MeshTriPoint
{
public:
    ~MeshTriPoint();
    ///< left face of this edge is considered
    MR::DotNet::EdgeId e;
    /// barycentric coordinates
    /// \details a in [0,1], a=0 => point is on next( e ) edge, a=1 => point is in dest( e )
    /// b in [0,1], b=0 => point is on e edge, b=1 => point is in dest( next( e ) )
    /// a+b in [0,1], a+b=0 => point is in org( e ), a+b=1 => point is on prev( e.sym() ) edge
    MR::DotNet::TriPoint bary;

internal:
    MeshTriPoint( MR::MeshTriPoint* mtp );
private:
    MR::MeshTriPoint* mtp_;
};

MR_DOTNET_NAMESPACE_END

