#pragma once

#include "MRPolylineTopology.h"
#include "MRUniqueThreadSafeOwner.h"
#include "MRPlane3.h"

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

    /// creates polyline from 2D contours, 3D polyline will get zero z-component
    MRMESH_API Polyline( const Contours2f& contours );
    /// creates polyline from 3D contours, 2D polyline will lose z-component
    MRMESH_API Polyline( const Contours3f& contours );

    /// adds connected line in this, passing progressively via points *[vs, vs+num)
    /// \details if closed argument is true then the last and the first points will be additionally connected
    /// \return the edge from first new to second new vertex    
    MRMESH_API EdgeId addFromPoints( const V * vs, size_t num, bool closed );

    /// adds connected line in this, passing progressively via points *[vs, vs+num)
    /// \details if vs[0] == vs[num-1] then a closed line is created
    /// \return the edge from first new to second new vertex
    MRMESH_API EdgeId addFromPoints( const V * vs, size_t num );

    /// appends polyline (from) in addition to this polyline: creates new edges, faces, verts and points
    MRMESH_API void addPartByMask( const Polyline<V>& from, const UndirectedEdgeBitSet& mask,
        VertMap* outVmap = nullptr, EdgeMap* outEmap = nullptr );

    /// returns coordinates of the edge origin
    V orgPnt( EdgeId e ) const { return points[ topology.org( e ) ]; }
    /// returns coordinates of the edge destination
    V destPnt( EdgeId e ) const { return points[ topology.dest( e ) ]; }
    /// returns a point on the edge: origin point for f=0 and destination point for f=1
    V edgePoint( EdgeId e, float f ) const { return f * destPnt( e ) + ( 1 - f ) * orgPnt( e ); }
    /// returns edge's centroid
    V edgeCenter( EdgeId e ) const { return edgePoint( e, 0.5f ); }

    /// returns vector equal to edge destination point minus edge origin point
    V edgeVector( EdgeId e ) const { return destPnt( e ) - orgPnt( e ); }
    /// returns Euclidean length of the edge
    float edgeLength( EdgeId e ) const { return edgeVector( e ).length(); }
    /// returns squared Euclidean length of the edge (faster to compute than length)
    float edgeLengthSq( EdgeId e ) const { return edgeVector( e ).lengthSq(); }
    /// returns total length of the polyline
    MRMESH_API float totalLength() const;

    /// returns cached aabb-tree for this polyline, creating it if it did not exist in a thread-safe manner
    MRMESH_API const AABBTreePolyline<V>& getAABBTree() const;
    /// returns cached aabb-tree for this polyline, but does not create it if it did not exist
    const AABBTreePolyline<V> * getAABBTreeNotCreate() const { return AABBTreeOwner_.get(); }

    /// returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())
    MRMESH_API Box<V> getBoundingBox() const;
    /// passes through all valid points and finds the minimal bounding box containing all of them
    /// \details if toWorld transformation is given then returns minimal bounding box in world space
    MRMESH_API Box<V> computeBoundingBox( const AffineXf<V> * toWorld = nullptr ) const;

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
    MRMESH_API Contours<V> contours() const;

    /// convert Polyline to simple 2D contour structures with vector of points inside
    /// \details if all even edges are consistently oriented, then the output contours will be oriented the same
    MRMESH_API Contours2f contours2() const;

    /// convert Polyline3 to Polyline2 or vice versa
    template<typename U>
    Polyline<U> toPolyline() const;

    /// adds path to this polyline
    /// \return the edge from first new to second new vertex
    MRMESH_API EdgeId addFromEdgePath( const Mesh& mesh, const EdgePath& path );

    /// adds path to this polyline
    /// \return the edge from first new to second new vertex
    MRMESH_API EdgeId addFromSurfacePath( const Mesh& mesh, const SurfacePath& path );

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;
    /// reflects the polyline from a given plane. Enabled only for Plane3f
    template < typename = std::enable_if_t< std::is_same_v<V, Vector3f> > >
    void mirror( const Plane3f& plane )
    {
        for ( auto& p : points )
        {
            p += 2.0f * ( plane.project( p ) - p );
        }
    }

private:
    mutable UniqueThreadSafeOwner<AABBTreePolyline<V>> AABBTreeOwner_;
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
