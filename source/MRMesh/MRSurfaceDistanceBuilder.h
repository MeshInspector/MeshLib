#pragma once

#include "MRId.h"
#include "MRVector.h"
#include "MRVector3.h"
#include <cfloat>
#include <optional>
#include <queue>

namespace MR
{

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
    /// initiates distance construction from given vertices with known start distance in all of them (region vertices will NOT be returned by growOne)
    MRMESH_API void addStartRegion( const VertBitSet & region, float startDistance );
    /// initiates distance construction from given start vertices with values in them (these vertices will NOT be returned by growOne if values in them are not decreased)
    MRMESH_API void addStartVertices( const HashMap<VertId, float>& startVertices );
    /// initiates distance construction from triangle vertices surrounding given start point (they all will be returned by growOne)
    MRMESH_API void addStart( const MeshTriPoint & start );
    /// processes one more candidate vertex, which is returned
    MRMESH_API VertId growOne();
    /// takes ownership over constructed distance map
    Vector<float,VertId> takeDistanceMap() { return std::move( vertDistanceMap_ ); }

public:
    /// returns true if further growth is impossible
    bool done() const { return nextVerts_.empty(); }
    /// returns path length till the next candidate vertex or maximum float value if all vertices have been reached
    float doneDistance() const { return nextVerts_.empty() ? FLT_MAX : nextVerts_.top().distance; }

private:
    const Mesh & mesh_;
    const VertBitSet* region_{nullptr};
    Vector<float,VertId> vertDistanceMap_;
    Vector<char,VertId> vertUpdatedTimes_;
    std::priority_queue<VertDistance> nextVerts_;
    std::optional<Vector3f> target_;

    /// compares proposed distance with the value known in c.vert
    /// \details if proposed distance is smaller then adds it in the queue and returns true; \n
    /// otherwise if the known distance to c.vert is already not greater than returns false
    bool suggestVertDistance_( VertDistance c );
    /// suggests new distance around a (recently updated) vertex
    void suggestDistancesAround_( VertId v );
    /// consider a path going in the left triangle from edge (e) to the opposing vertex
    void considerLeftTriPath_( EdgeId e );
    /// if target is provided then adds distance to it to the metric
    float metricToPenalty_( float metric, VertId v ) const;
};

/// \}

} // namespace MR
