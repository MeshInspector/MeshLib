#pragma once
#include "MRMeshFwd.h"

namespace MR
{
/// This function splits edges intersected by the plane
/// \return edges located above the plane (in direction of normal to plane)
/// \param polyline Input polyline that will be cut by the plane
/// \param plane Input plane to cut polyline with
/// \param newPositiveEdges edges with origin on the plane and oriented to the positive direction (only adds bits to the existing ones)
/// \param onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
MRMESH_API UndirectedEdgeBitSet subdivideWithPlane( Polyline3& polyline, const Plane3f& plane, EdgeBitSet* newPositiveEdges = {}, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback = {} );
[[deprecated]] MRMESH_API UndirectedEdgeBitSet subdividePolylineWithPlane( Polyline3& polyline, const Plane3f& plane, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback = {} );

struct DividePolylineParameters
{
    /// onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
    std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback;
    /// closeLineAfterCut if true, the ends of resulting polyline will be connected by new edges (can make a polyline closed, even if the original one was open)
    /// if close, only cut edges (no new edges will be created)
    bool closeLineAfterCut = false;
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

/// This function divides polyline with a plane, leaving only part of polyline that lies in positive direction of normal
/// \param polyline Input polyline that will be cut by the plane
/// \param plane Input plane to cut polyline with
/// \param params parameters of the function, containing optional output
MRMESH_API void trimWithPlane( Polyline3& polyline, const Plane3f& plane, const DividePolylineParameters& params = {} );
[[deprecated]] MRMESH_API void dividePolylineWithPlane( Polyline3& polyline, const Plane3f& plane, const DividePolylineParameters& params = {} );

/// This function cuts polyline with a plane
/// \return Edge segments that are closer to the plane than \param eps
/// \param polyline Input polyline that will be cut by the plane
/// \param plane Input plane to cut polyline with
/// \param eps Maximal distance from the plane
/// \param edges in a positive half-space relative to the plane (only adds bits to the existing ones)
MRMESH_API std::vector<EdgeSegment> extractSectionsFromPolyline( const Polyline3& polyline, const Plane3f& plane, float eps, UndirectedEdgeBitSet* positiveEdges = {} );

}