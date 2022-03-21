#pragma once

#include "MRPolylineTopology.h"
#include "MRUniqueThreadSafeOwner.h"
#include "MRVector2.h"

namespace MR
{

// 2 dimensional polyline
struct Polyline2
{
public:
    PolylineTopology topology;
    Vector<Vector2f, VertId> points;

    Polyline2() = default;

    // creates polyline from 2d contours
    MRMESH_API Polyline2( const Contours2f& contours );

    // adds connected line in this, passing progressively via points *[vs, vs+num);
    // if closed argument is true then the last and the first points will be additionally connected;
    // return the edge from first new to second new vertex    
    MRMESH_API EdgeId addFromPoints( const Vector2f* vs, size_t num, bool closed );
    [[deprecated]] EdgeId makePolyline( const Vector2f* vs, size_t num, bool closed ) { return addFromPoints( vs, num, closed ); }

    // adds connected line in this, passing progressively via points *[vs, vs+num)
    // if vs[0] == vs[num-1] then a closed line is created;
    // return the edge from first new to second new vertex
    MRMESH_API EdgeId addFromPoints( const Vector2f * vs, size_t num );
    [[deprecated]] EdgeId makePolyline( const Vector2f * vs, size_t num ) { return addFromPoints( vs, num ); }

    // returns coordinates of the edge origin
    Vector2f orgPnt( EdgeId e ) const { return points[ topology.org( e ) ]; }
    // returns coordinates of the edge destination
    Vector2f destPnt( EdgeId e ) const { return points[ topology.dest( e ) ]; }
    // returns a point on the edge: origin point for f=0 and destination point for f=1
    Vector2f edgePoint( EdgeId e, float f ) const { return f * destPnt( e ) + ( 1 - f ) * orgPnt( e ); }

    // returns vector equal to edge destination point minus edge origin point
    Vector2f edgeVector( EdgeId e ) const { return destPnt( e ) - orgPnt( e ); }
    // returns Euclidean length of the edge
    float edgeLength( EdgeId e ) const { return edgeVector( e ).length(); }
    // returns squared Euclidean length of the edge (faster to compute than length)
    float edgeLengthSq( EdgeId e ) const { return edgeVector( e ).lengthSq(); }
    // returns total length of the polyline
    MRMESH_API float totalLength() const;

    // returns cached aabb-tree for this polyline, creating it if it did not exist in a thread-safe manner
    MRMESH_API const AABBTreePolyline2& getAABBTree() const;
    // returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())
    MRMESH_API Box2f getBoundingBox() const;

    // applies given transformation to all valid polyline vertices
    MRMESH_API void transform( const AffineXf2f & xf );

    // Invalidates caches (e.g. aabb-tree) after a change in polyline
    void invalidateCaches() { AABBTreeOwner_.reset(); };

    // convert Polyline2 to simple contour structures with vector of points inside
    // if all even edges are consistently oriented, then the output contours will be oriented the same
    MRMESH_API Contours2f contours() const;

    //convert Polyline2 to 3D Polyline3 by adding zeros for a Z components
    MRMESH_API Polyline3 toPolyline3() const;

private:
    mutable UniqueThreadSafeOwner<AABBTreePolyline2> AABBTreeOwner_;
};

} //namespace MR
