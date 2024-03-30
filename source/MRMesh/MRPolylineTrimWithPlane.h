#pragma once
#pragma once
#include "MRMeshFwd.h"

namespace MR
{
/// This function splits edges intersected by the plane
/// \return New edges with origin on the plane and oriented to the positive direction
/// \param polyline Input polyline that will be cut by the plane
/// \param plane Input plane to cut polyline with
/// \param onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
MRMESH_API EdgeBitSet subdividePolylineWithPlane( Polyline3& polyline, const Plane3f& plane, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback = nullptr );

struct DividePolylineParameters
{
    /// onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
    std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback;
    /// fillAfterCut if true, the ends of resulting polyline will be united with new edges
    bool fillAfterCut = false;
    /// map from input polyline verts to output
    VertMap* outVmap = nullptr;
    /// map from input polyline edges to output
    EdgeMap* outEmap = nullptr;
    /// otherPart Optional return, polyline composed from edges on the negative side of the plane
    Polyline3* otherPart = nullptr;
    ///  map from input polyline verts to other output
    VertMap* otherOutVmap = nullptr;
    /// map from input polyline edges to other output
    EdgeMap* otherOutEmap = nullptr;
};

/// This function divides polyline with a plane
/// \param polyline Input polyline that will be cut by the plane
/// \param plane Input plane to cut polyline with
/// \param params parameters of the function, contains optional output
MRMESH_API void dividePolylineWithPlane( Polyline3& polyline, const Plane3f& plane, const DividePolylineParameters& params = {} );

/// This function cuts polyline with a plane
/// \return Edge segments tha are closer to the plane than \param eps
/// \param polyline Input polyline that will be cut by the plane
/// \param plane Input plane to cut polyline with
/// \param eps Maximal distance from the plane
MRMESH_API std::vector<EdgeSegment> extractSectionsFromPolyline( const Polyline3& polyline, const Plane3f& plane, float eps );

}