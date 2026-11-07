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
    /// the vertices connected by mesh edges make lines or regions, each isolated vertex is a point
    MRMESH_API void addStartVertices( const HashMap<VertId, float>& startVertices );
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
    /// otherwise its distance is from the line through vertSource_ and this start point (a start edge of a line, or any start point near a region)
    Vector<SurfaceDistanceStartId, VertId> vertLineEnd_;
    /// for each start point: the distance assigned to it in the beginning
    Vector<float, SurfaceDistanceStartId> sourceStartDistance_;
    /// for each start point: its vertex, invalid for a start inside a triangle or on an edge
    Vector<VertId, SurfaceDistanceStartId> sourceVert_;
    /// groups of start points connected by mesh edges, the parent of each start point is the root of its group;
    /// the field is assumed linear only between start points of one group (along a line or in a region)
    BaseUnionFind<SurfaceDistanceStartId> sourceGroup_;
    /// start points of the groups making regions (some start point has more than three start neighbors), not lines;
    /// near a region the distances are not split between start points and start edges, but taken from the linear field
    TypedBitSet<SurfaceDistanceStartId> regionSources_;

    /// compares proposed distance with the value known in c.vert
    /// \details if proposed distance is smaller then adds it in the queue and returns true; \n
    /// otherwise if the known distance to c.vert is already not greater than returns false
    bool suggestVertDistance_( VertDistance c, SurfaceDistanceStartId source, SurfaceDistanceStartId lineEnd );
    /// creates new start point with given initial distance and its vertex (if any)
    SurfaceDistanceStartId addSource_( float startDistance, VertId v );
    /// given point p near the line through start points s0 and s1 of one group, replaces them with the nearest start edge
    /// or with the nearest start point (and invalid s1) if p is beyond the ends of the start edges
    void nearestStartEdge_( const Vector3f & p, SurfaceDistanceStartId & s0, SurfaceDistanceStartId & s1 ) const;
    /// estimates the distance from reached vertex v to start point s0 (if s1 is invalid) or to the line through start edge s0-s1,
    /// using the distance stored in v and the geometry near the start points; returns false if v is not near them
    bool distToStart_( VertId v, SurfaceDistanceStartId s0, SurfaceDistanceStartId s1, double & d ) const;
    /// unites the group of the start point in v with the groups of start points in its neighbor vertices
    void uniteTouchingSources_( VertId v );
    /// finds the groups of start points, then fills regionSources_
    void finishSourceGroups_();
    /// suggests new distance around a (recently updated) vertex
    void suggestDistancesAround_( VertId v );
    /// consider a path going in the left triangle from edge (e) to the opposing vertex
    void considerLeftTriPath_( EdgeId e );
    /// computes the distance to c in triangle abc, where a and b have distances (va <= vb) from different start points of one group,
    /// or from one start point and from the linear field near it
    void considerTriNearStarts_( VertId a, VertId b, VertId c, float va, float vb );
    /// if target is provided then adds distance to it to the metric
    float metricToPenalty_( float metric, VertId v ) const;
};

/// consider triangle 0bc, where a linear scalar field is defined in two points: v(0) = 0, v(b) = b;
/// computes the field in c-point;
/// returns false if field gradient enters c-point not from inside of the triangle
[[nodiscard]] MRMESH_API bool getFieldAtC( const Vector3f & b, const Vector3f & c, float vb, float & vc );

/// \}

} // namespace MR
