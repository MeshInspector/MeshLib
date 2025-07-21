#pragma once

#include "MRPolylineTopology.h"
#include "MRSharedThreadSafeOwner.h"
#include "MRPlane3.h"
#include "MRLineSegm.h"
#include "MREdgePoint.h"

namespace MR
{

/// \defgroup PolylineGroup Polyline

/// polyline that stores points of type V
/// \ingroup PolylineGroup
template<typename V>
struct Polyline
{
public:
    PolylineTopology topology;
    Vector<V, VertId> points;

    Polyline() = default;

    /// creates polyline from one contour (open or closed)
    MRMESH_API Polyline( const Contour<V>& contour );

    /// creates polyline from several contours (each can be open or closed)
    MRMESH_API Polyline( const Contours<V>& contours );

    /// creates comp2firstVert.size()-1 not-closed polylines
    /// each pair (a,b) of indices in \param comp2firstVert defines vertex range of a polyline [a,b)
    /// \param ps point coordinates
    MRMESH_API Polyline( const std::vector<VertId> & comp2firstVert, Vector<V, VertId> ps );

    /// adds connected line in this, passing progressively via points *[vs, vs+num)
    /// \details if closed argument is true then the last and the first points will be additionally connected
    /// \return the edge from first new to second new vertex    
    MRMESH_API EdgeId addFromPoints( const V * vs, size_t num, bool closed );

    /// adds connected line in this, passing progressively via points *[vs, vs+num)
    /// \details if num > 2 && vs[0] == vs[num-1] then a closed line is created
    /// \return the edge from first new to second new vertex
    MRMESH_API EdgeId addFromPoints( const V * vs, size_t num );

    /// appends polyline (from) in addition to this polyline: creates new edges, verts and points;
    /// \param outVmap,outEmap (optionally) returns mappings: from.id -> this.id
    MRMESH_API void addPart( const Polyline<V>& from,
        VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr );

    /// appends polyline (from) in addition to this polyline: creates new edges, verts and points
    MRMESH_API void addPartByMask( const Polyline<V>& from, const UndirectedEdgeBitSet& mask,
        VertMap* outVmap = nullptr, EdgeMap* outEmap = nullptr );

    /// tightly packs all arrays eliminating lone edges and invalid verts and points,
    /// optionally returns mappings: old.id -> new.id
    MRMESH_API void pack( VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr );

    /// returns coordinates of the edge origin
    [[nodiscard]] V orgPnt( EdgeId e ) const { return points[ topology.org( e ) ]; }

    /// returns coordinates of the edge destination
    [[nodiscard]] V destPnt( EdgeId e ) const { return points[ topology.dest( e ) ]; }

    /// returns a point on the edge: origin point for f=0 and destination point for f=1
    [[nodiscard]] V edgePoint( EdgeId e, float f ) const { return f * destPnt( e ) + ( 1 - f ) * orgPnt( e ); }

    /// computes coordinates of point given as edge and relative position on it
    [[nodiscard]] V edgePoint( const EdgePoint & ep ) const { return edgePoint( ep.e, ep.a ); }

    /// returns edge's centroid
    [[nodiscard]] V edgeCenter( EdgeId e ) const { return edgePoint( e, 0.5f ); }

    /// returns vector equal to edge destination point minus edge origin point
    [[nodiscard]] V edgeVector( EdgeId e ) const { return destPnt( e ) - orgPnt( e ); }

    /// returns line segment of given edge
    [[nodiscard]] LineSegm<V> edgeSegment( EdgeId e ) const { return LineSegm<V>( orgPnt( e ), destPnt( e ) ); }

    /// converts vertex into edge-point representation
    [[nodiscard]] EdgePoint toEdgePoint( VertId v ) const { return EdgePoint( topology, v ); }

    /// converts edge and point's coordinates into edge-point representation
    [[nodiscard]] MRMESH_API EdgePoint toEdgePoint( EdgeId e, const V & p ) const;

    /// returns Euclidean length of the edge
    [[nodiscard]] float edgeLength( EdgeId e ) const { return edgeVector( e ).length(); }

    /// returns squared Euclidean length of the edge (faster to compute than length)
    [[nodiscard]] float edgeLengthSq( EdgeId e ) const { return edgeVector( e ).lengthSq(); }

    /// calculates directed loop area if iterating in `e` direction
    /// .z = FLT_MAX if `e` does not represent a loop
    [[nodiscard]] MRMESH_API Vector3f loopDirArea( EdgeId e ) const;

    /// returns total length of the polyline
    [[nodiscard]] MRMESH_API float totalLength() const;

    /// returns average edge length in the polyline
    [[nodiscard]] float averageEdgeLength() const { auto n = topology.computeNotLoneUndirectedEdges(); return n ? totalLength() / n : 0.0f; }

    /// returns cached aabb-tree for this polyline, creating it if it did not exist in a thread-safe manner
    MRMESH_API const AABBTreePolyline<V>& getAABBTree() const;

    /// returns cached aabb-tree for this polyline, but does not create it if it did not exist
    [[nodiscard]] const AABBTreePolyline<V> * getAABBTreeNotCreate() const { return AABBTreeOwner_.get(); }

    /// returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())
    [[nodiscard]] MRMESH_API Box<V> getBoundingBox() const;

    /// passes through all valid points and finds the minimal bounding box containing all of them
    /// \details if toWorld transformation is given then returns minimal bounding box in world space
    [[nodiscard]] MRMESH_API Box<V> computeBoundingBox( const AffineXf<V> * toWorld = nullptr ) const;

    // computes average position of all valid polyline vertices
    [[nodiscard]] MRMESH_API V findCenterFromPoints() const;

    /// applies given transformation to all valid polyline vertices
    MRMESH_API void transform( const AffineXf<V> & xf );

    /// split given edge on two parts:
    /// dest(returned-edge) = org(e) - newly created vertex,
    /// org(returned-edge) = org(e-before-split),
    /// dest(e) = dest(e-before-split)
    MRMESH_API EdgeId splitEdge( EdgeId e, const V & newVertPos );

    // same, but split given edge on two equal parts
    EdgeId splitEdge( EdgeId e ) { return splitEdge( e, edgeCenter( e ) ); }

    /// Invalidates caches (e.g. aabb-tree) after a change in polyline
    void invalidateCaches() { AABBTreeOwner_.reset(); };

    /// convert Polyline to simple contour structures with vector of points inside
    /// \details if all even edges are consistently oriented, then the output contours will be oriented the same
    /// \param vertMap optional output map for for each contour point to corresponding VertId
    [[nodiscard]] MRMESH_API Contours<V> contours( std::vector<std::vector<VertId>>* vertMap = nullptr ) const;

    /// convert Polyline3 to Polyline2 or vice versa
    template<typename U>
    [[nodiscard]] Polyline<U> toPolyline() const;

    /// adds path to this polyline
    /// \return the edge from first new to second new vertex
    MRMESH_API EdgeId addFromEdgePath( const Mesh& mesh, const EdgePath& path );

    /// adds path to this polyline
    /// \return the edge from first new to second new vertex
    EdgeId addFromSurfacePath( const Mesh& mesh, const SurfacePath& path ) { return addFromGeneralSurfacePath( mesh, {}, path, {} ); }

    /// adds general path = start-path-end (where both start and end are optional) to this polyline
    /// \return the edge from first new to second new vertex
    MRMESH_API EdgeId addFromGeneralSurfacePath( const Mesh& mesh, const MeshTriPoint & start, const SurfacePath& path, const MeshTriPoint & end );

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;
    /// reflects the polyline from a given plane. Enabled only for Polyline3f
    template <class Q = V>
    [[nodiscard]] typename std::enable_if_t< std::is_same_v<Q, Vector3f> > mirror( const Plane3f& plane )
    {
        for ( auto& p : points )
        {
            p += 2.0f * ( plane.project( p ) - p );
        }

        invalidateCaches();
    }

private:
    mutable SharedThreadSafeOwner<AABBTreePolyline<V>> AABBTreeOwner_;
};

template<typename V>
template<typename U>
Polyline<U> Polyline<V>::toPolyline() const
{
    Polyline<U> res;
    res.topology = topology;
    res.points.reserve( points.size() );
    for ( size_t i = 0; i < points.size(); i++ )
    {
        res.points.push_back( U{ points[VertId( i )] } );
    }
    return res;
}

} // namespace MR
