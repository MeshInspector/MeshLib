#pragma once

#include "MRBitSet.h"
#include "MRId.h"
#include "MRUnionFind.h"
#include "MRVector.h"
#include "MRVector3.h"
#include <cfloat>
#include <optional>
#include <queue>

namespace MR
{

class SurfaceDistanceStartTag;
/// identifier of a start point in SurfaceDistanceBuilder
using SurfaceDistanceStartId = Id<SurfaceDistanceStartTag>;

/// \defgroup SurfaceDistanceBuilder Group Surface Distance Builder
/// \ingroup SurfacePathGroup
/// \{

struct VertDistance
{
    /// vertex in question
    VertId vert;
    /// best known distance to reach this vertex
    float distance = FLT_MAX;

    VertDistance() = default;
    VertDistance( VertId v, float d ) : vert( v ), distance( d ) { }
};

/// smaller distance to be the first
inline bool operator <( const VertDistance & a, const VertDistance & b )
{
    return a.distance > b.distance;
}

/// this class is responsible for iterative construction of distance map along the surface
class SurfaceDistanceBuilder
{
public:
    MRMESH_API SurfaceDistanceBuilder( const Mesh & mesh, const VertBitSet* region );
    SurfaceDistanceBuilder( const Mesh & mesh, const Vector3f & target, const VertBitSet* region ) :
        SurfaceDistanceBuilder( mesh, region) { target_ = target; }
    /// initiates distance construction from given vertices with known start distance in all of them (region vertices will NOT be returned by growOne);
    /// the vertices connected by mesh edges make lines or regions, each isolated vertex is a point
    MRMESH_API void addStartRegion( const VertBitSet & region, float startDistance );
    /// initiates distance construction from given start vertices with values in them (these vertices will NOT be returned by growOne if values in them are not decreased);
    /// the vertices connected by mesh edges make lines or regions (a line with different values is handled as a region), each isolated vertex is a point
    MRMESH_API void addStartVertices( const HashMap<VertId, float>& startVertices );
    /// initiates distance construction from given start edges and start points with known start distance in all of them (their vertices will NOT be returned by growOne);
    /// the start edges connected by vertices make lines or regions; each start point is a point never united with other starts,
    /// even if it is one edge away from them, unless it is an end of a start edge
    MRMESH_API void addStartEdgesAndPoints( const UndirectedEdgeBitSet & startEdges, const VertBitSet & startPoints, float startDistance );
    /// initiates distance construction from triangle vertices surrounding given start point (they all will be returned by growOne);
    /// the distances are from this point, each call adds a separate point
    MRMESH_API void addStart( const MeshTriPoint & start );

    /// the maximum amount of times vertex distance can be updated in [1,255], 3 by default;
    /// the more the better obtuse triangles are handled
    MRMESH_API void setMaxVertUpdates( int v );

    /// processes one more candidate vertex, which is returned
    MRMESH_API VertId growOne();
    /// takes ownership over constructed distance map
    VertScalars takeDistanceMap() { return std::move( vertDistanceMap_ ); }

public:
    /// returns true if further growth is impossible
    bool done() const { return nextVerts_.empty(); }
    /// returns path length till the next candidate vertex or maximum float value if all vertices have been reached
    float doneDistance() const { return nextVerts_.empty() ? FLT_MAX : nextVerts_.top().distance; }

private:
    const Mesh & mesh_;
    const VertBitSet* region_{nullptr};
    VertScalars vertDistanceMap_;
    Vector<char,VertId> vertUpdatedTimes_;
    std::priority_queue<VertDistance> nextVerts_;
    std::optional<Vector3f> target_;
    int maxVertUpdates_ = 3;

    /// for each reached vertex: the start point, the distance from which is stored in the vertex
    Vector<SurfaceDistanceStartId, VertId> vertSource_;
    /// for each reached vertex: invalid if its distance is from the point vertSource_,
    /// vertSource_ itself if its distance is from the linear field near a region,
    /// otherwise its distance is from the start edge of a line between vertSource_ and this start point
    Vector<SurfaceDistanceStartId, VertId> vertLineEnd_;
    /// for each start point: the distance assigned to it in the beginning
    Vector<float, SurfaceDistanceStartId> sourceStartDistance_;
    /// for each start point: its vertex, invalid for a start inside a triangle or on an edge
    Vector<VertId, SurfaceDistanceStartId> sourceVert_;
    /// for each start point: its position
    Vector<Vector3f, SurfaceDistanceStartId> sourcePos_;
    /// the mesh edges between start points of one group
    UndirectedEdgeBitSet startEdges_;
    /// groups of start points connected by start edges, the parent of each start point is the root of its group;
    /// the field is assumed linear only between start points of one group (along a line or in a region)
    BaseUnionFind<SurfaceDistanceStartId> sourceGroup_;
    /// start points of the groups making regions (some start point has more than three start neighbors), not lines,
    /// and of the groups with different start distances;
    /// near a region the distances are not split between start points and start edges, but taken from the linear field
    TypedBitSet<SurfaceDistanceStartId> regionSources_;

    /// compares proposed distance with the value known in c.vert
    /// \details if proposed distance is smaller then adds it in the queue and returns true; \n
    /// otherwise if the known distance to c.vert is already not greater than returns false
    bool suggestVertDistance_( VertDistance c, SurfaceDistanceStartId source, SurfaceDistanceStartId lineEnd );
    /// creates new start point with given initial distance, its vertex (if any) and position
    SurfaceDistanceStartId addSource_( float startDistance, VertId v, const Vector3f & pos );
    /// makes v a start point if its distance is larger than given one
    void addStartVert_( VertId v, float startDistance );
    /// returns the start point in v if v is a start vertex, otherwise invalid id
    SurfaceDistanceStartId startInVert_( VertId v ) const;
    /// if both ends of e are start vertices, then marks e as a start edge and unites the groups of its ends
    void addStartEdge_( EdgeId e );
    /// given point p near the line through start points s0 and s1 of one group, replaces them with the nearest start edge
    /// or with the nearest start point (and invalid s1) if p is beyond the ends of the start edges
    void nearestStartEdge_( const Vector3f & p, SurfaceDistanceStartId & s0, SurfaceDistanceStartId & s1 ) const;
    /// estimates the distance from reached vertex v to start point s0 (if s1 is invalid) or to the line through start edge s0-s1,
    /// using the distance stored in v and the geometry near the start points; returns false if v is not near them
    bool distToStart_( VertId v, SurfaceDistanceStartId s0, SurfaceDistanceStartId s1, double & d ) const;
    /// given the distances da and db (relative to the start distance) from a and b to start point s0 (if s1 is invalid) or to the line of start edge s0-s1,
    /// computes the distance to c in triangle abc (with the start distance); returns false if c is not beside the start edge,
    /// or (if checkCross) the straight path from the start point to c does not cross ab
    bool distAtC_( const Vector3f & pa, const Vector3f & pb, const Vector3f & pc, SurfaceDistanceStartId s0, SurfaceDistanceStartId s1,
        double da, double db, double & vc, bool checkCross = true ) const;
    /// returns the straight-line distance (relative to the start distance) from p to start point s0 or to start edge s0-s1
    /// (or to the line through it), which is not larger than the distance along the surface
    double straightDistToStart_( const Vector3f & p, SurfaceDistanceStartId s0, SurfaceDistanceStartId s1, bool infiniteLine = false ) const;
    /// finds the groups of start points, then fills regionSources_
    void finishSourceGroups_();
    /// suggests new distance around a (recently updated) vertex
    void suggestDistancesAround_( VertId v );
    /// consider a path going in the left triangle from edge (e) to the opposing vertex
    void considerLeftTriPath_( EdgeId e );
    /// computes the distance to c in left triangle abc of edge e, where a and b have distances (va <= vb) from different start points of one group,
    /// or from one start point and from the linear field near it
    void considerTriNearStarts_( EdgeId e, VertId a, VertId b, VertId c, float va, float vb );
    /// computes the distance to c in left triangle abc of edge e, where a and b have distances from starts of different groups:
    /// the field of each start is extended to the other vertex through the triangle on the other side of e
    void considerTriBetweenGroups_( EdgeId e, VertId a, VertId b, VertId c, float va );
    /// if target is provided then adds distance to it to the metric
    float metricToPenalty_( float metric, VertId v ) const;
};

/// consider triangle 0bc, where a linear scalar field is defined in two points: v(0) = 0, v(b) = b;
/// computes the field in c-point;
/// returns false if field gradient enters c-point not from inside of the triangle
[[nodiscard]] MRMESH_API bool getFieldAtC( const Vector3f & b, const Vector3f & c, float vb, float & vc );

/// \}

} // namespace MR
