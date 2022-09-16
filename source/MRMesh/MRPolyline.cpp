#include "MRPolyline.h"
#include "MRPolylineEdgeIterator.h"
#include "MRAABBTreePolyline.h"
#include "MRAffineXf2.h"
#include "MRAffineXf3.h"
#include "MRVector2.h"
#include "MRGTest.h"
#include "MRTimer.h"
#include "MRMesh.h"
#include "MRComputeBoundingBox.h"
#include "MRPch/MRTBB.h"

namespace MR
{

template<typename V>
Polyline<V>::Polyline( const Contours2f& contours )
{
    MR_TIMER
    topology.buildFromContours( contours, 
        [&points = this->points]( size_t sz )
        {
            points.reserve( sz );
        },
        [&points = this->points]( const Vector2f & p )
        {
            if constexpr ( V::elements == 2 )
                points.emplace_back( p.x, p.y );
            else
                points.emplace_back( p.x, p.y, 0.0f );
            return points.backId();
        } 
    );
}

template<typename V>
Polyline<V>::Polyline( const Contours3f& contours )
{
    MR_TIMER
    topology.buildFromContours( contours, 
        [&points = this->points]( size_t sz )
        {
            points.reserve( sz );
        },
        [&points = this->points]( const Vector3f & p )
        {
            if constexpr ( V::elements == 2 )
                points.emplace_back( p.x, p.y );
            else
                points.push_back( p );
            return points.backId();
        } 
    );
}

template<typename V>
EdgeId Polyline<V>::addFromPoints( const V * vs, size_t num, bool closed )
{
    if ( !vs || num < 2 )
    {
        assert( false );
        return {};
    }
    const VertId firstVertId( (int)topology.vertSize() );
    if ( (int)firstVertId + num > points.size() )
        points.resize( (int)firstVertId + num );

    const size_t numSegmEnds = num + ( closed ? 1 : 0 );
    std::vector<VertId> newVerts( numSegmEnds );
    for ( int i = 0; i < num; ++i )
    {
        VertId v( firstVertId + i );
        newVerts[i] = v;
        points[v] = vs[i];
    }
    if ( closed )
        newVerts.back() = newVerts.front();

    auto e = topology.makePolyline( newVerts.data(), numSegmEnds );
    invalidateCaches();
    return e;
}

template<typename V>
EdgeId Polyline<V>::addFromPoints( const V * vs, size_t num )
{
    if ( !vs || num < 2 )
    {
        assert( false );
        return {};
    }
    const bool closed = vs[0] == vs[num-1];
    return addFromPoints( vs, num - ( closed ? 1 : 0 ), closed );
}

template<typename V>
void MR::Polyline<V>::addPartByMask( const Polyline<V>& from, const UndirectedEdgeBitSet& mask, 
    VertMap* outVmap /*= nullptr*/, EdgeMap* outEmap /*= nullptr */ )
{
    MR_TIMER;

    VertMap vmap;
    VertMap* vmapPtr = outVmap ? outVmap : &vmap;
    topology.addPartByMask( from.topology, mask, vmapPtr, outEmap );
    const VertMap& vmapRef = *vmapPtr;

    VertId lastPointId = topology.lastValidVert();
    if ( points.size() < lastPointId + 1 )
        points.resize( lastPointId + 1 );

    for ( VertId fromv{ 0 }; fromv < vmapRef.size(); ++fromv )
    {
        VertId v = vmapRef[fromv];
        if ( v.valid() )
            points[v] = from.points[fromv];
    }

    invalidateCaches();
}


template<typename V>
float Polyline<V>::totalLength() const
{
    MR_TIMER
    double sum = 0;
    for ( auto ue : undirectedEdges( topology ) )
        sum += edgeLength( ue );

    return (float)sum;
}

template<typename V>
Box<V> Polyline<V>::getBoundingBox() const
{
    return getAABBTree().getBoundingBox();
}

template<typename V>
Box<V> Polyline<V>::computeBoundingBox( const AffineXf<V> * toWorld ) const
{
    return MR::computeBoundingBox( points, topology.getValidVerts(), toWorld );
}

template<typename V>
Contours<V> Polyline<V>::contours() const
{
    MR_TIMER
    return topology.convertToContours<V>( 
        [&points = this->points]( VertId v )
        {
            return points[v];
        } 
    );
}

template<typename V>
Contours2f Polyline<V>::contours2() const
{
    MR_TIMER
    return topology.convertToContours<Vector2f>( 
        [&points = this->points]( VertId v )
        {
            return Vector2f{ points[v] };
        } 
    );
}

template<typename V>
EdgeId Polyline<V>::addFromEdgePath( const Mesh& mesh, const EdgePath& path )
{
    if ( path.empty() )
        return {};
    bool closed = mesh.topology.org( path.front() ) == mesh.topology.dest( path.back() );
    auto shift = points.size();
    points.resize( shift + path.size() + ( closed ? 0 : 1 ) );
    std::vector<VertId> newVerts( path.size() + 1 );
    for ( int i = 0; i < path.size(); ++i )
    {
        VertId newV = VertId( shift + i );
        newVerts[i] = newV;
        points[newV] = V{ mesh.orgPnt( path[i] ) };
    }
    if ( !closed )
    {
        newVerts.back() = VertId( shift + path.size() );
        points.back() = V{ mesh.destPnt( path.back() ) };
    }
    else
    {
        newVerts.back() = newVerts.front();
    }

    auto e = topology.makePolyline( newVerts.data(), newVerts.size() );
    invalidateCaches();
    return e;
}

template<typename V>
EdgeId Polyline<V>::addFromSurfacePath( const Mesh& mesh, const SurfacePath& path )
{
    if ( path.empty() )
        return {};
    bool closed = path.front() == path.back();
    auto shift = points.size();
    points.resize( shift + path.size() + ( closed ? -1 : 0 ) );
    std::vector<VertId> newVerts( path.size() );
    for ( int i = 0; i + 1 < path.size(); ++i )
    {
        VertId newV = VertId( shift + i );
        newVerts[i] = newV;
        points[newV] = V{ mesh.edgePoint( path[i] ) };
    }
    if ( !closed )
    {
        newVerts.back() = VertId( shift + path.size() - 1 );
        points.back() = V{ mesh.edgePoint( path.back() ) };
    }
    else
    {
        newVerts.back() = newVerts.front();
    }

    auto e = topology.makePolyline( newVerts.data(), newVerts.size() );
    invalidateCaches();
    return e;
}

template<typename V>
void Polyline<V>::transform( const AffineXf<V> & xf )
{
    MR_TIMER
    VertId lastValidVert = topology.lastValidVert();

    tbb::parallel_for(tbb::blocked_range<VertId>(VertId{ 0 }, lastValidVert + 1), [&](const tbb::blocked_range<VertId> & range)
    {
        for (VertId v = range.begin(); v < range.end(); ++v)
        {
            if (topology.hasVert(v))
                points[v] = xf(points[v]);
        }
    });
    invalidateCaches();
}

template<typename V>
EdgeId Polyline<V>::splitEdge( EdgeId e, const V & newVertPos )
{
    EdgeId newe = topology.splitEdge( e );
    points.autoResizeAt( topology.org( e ) ) = newVertPos;
    return newe;
}

template<typename V>
const AABBTreePolyline<V>& Polyline<V>::getAABBTree() const
{
    return AABBTreeOwner_.getOrCreate( [this]{ return AABBTreePolyline<V>( *this ); } );
}

template<typename V>
size_t Polyline<V>::heapBytes() const
{
    return topology.heapBytes()
        + points.heapBytes()
        + AABBTreeOwner_.heapBytes();
}

template struct Polyline<Vector2f>;
template struct Polyline<Vector3f>;

TEST( MRMesh, Polyline2 )
{
    Contour2f cont;
    cont.push_back( Vector2f( 0.f, 0.f ) );
    cont.push_back( Vector2f( 1.f, 0.f ) );
    cont.push_back( Vector2f( 0.f, 1.f ) );
    cont.push_back( Vector2f( 1.f, 1.f ) );

    Contour2f cont2;
    cont2.push_back( Vector2f( 2.f, 0.f ) );
    cont2.push_back( Vector2f( 3.f, 0.f ) );
    cont2.push_back( Vector2f( 2.f, 1.f ) );
    cont2.push_back( Vector2f( 3.f, 1.f ) );

    Contours2f conts{ cont,cont2 };

    Polyline2 pl( conts );
    auto conts2 = pl.contours();

    for ( auto i = 0; i < conts.size(); i++ )
{
        auto& c1 = conts[i];
        auto& c2 = conts2[i];
        for ( auto j = 0; j < c1.size(); j++ )
{
            auto v1 = c1[j];
            auto v2 = c2[j];
            EXPECT_NEAR( v1[0], v2[0], 1e-8 );
            EXPECT_NEAR( v1[1], v2[1], 1e-8 );
        }
    }
}

TEST( MRMesh, Polyline3 )
{
    Contour2f cont;
    cont.push_back( Vector2f( 0.f, 0.f ) );
    cont.push_back( Vector2f( 1.f, 0.f ) );
    cont.push_back( Vector2f( 0.f, 1.f ) );
    cont.push_back( Vector2f( 1.f, 1.f ) );

    Contour2f cont2;
    cont2.push_back( Vector2f( 2.f, 0.f ) );
    cont2.push_back( Vector2f( 3.f, 0.f ) );
    cont2.push_back( Vector2f( 2.f, 1.f ) );
    cont2.push_back( Vector2f( 3.f, 1.f ) );

    Contours2f conts{ cont,cont2 };

    Polyline3 pl( conts );
    auto conts2 = pl.contours();

    for ( auto i = 0; i < conts.size(); i++ )
    {
        auto& c1 = conts[i];
        auto& c2 = conts2[i];
        for ( auto j = 0; j < c1.size(); j++ )
        {
            auto v1 = c1[j];
            auto v2 = c2[j];
            EXPECT_NEAR( v1[0], v2[0], 1e-8 );
            EXPECT_NEAR( v1[1], v2[1], 1e-8 );
        }
    }
}

TEST( MRMesh, PolylineSplitEdge )
{
    Contour2f cont;
    cont.push_back( Vector2f( 0.f, 0.f ) );
    cont.push_back( Vector2f( 1.f, 0.f ) );
    Polyline2 polyline( { cont } );

    EXPECT_EQ( polyline.topology.numValidVerts(), 2 );
    EXPECT_EQ( polyline.points.size(), 2 );
    EXPECT_EQ( polyline.topology.lastNotLoneEdge(), EdgeId(1) ); // 1*2 = 2 half-edges in total

    auto e01 = polyline.topology.findEdge( 0_v, 1_v );
    EXPECT_TRUE( e01.valid() );
    auto ex = polyline.splitEdge( e01 );
    VertId v01 = polyline.topology.org( e01 );
    EXPECT_EQ( polyline.topology.dest( ex ), v01 );
    EXPECT_EQ( polyline.topology.numValidVerts(), 3 );
    EXPECT_EQ( polyline.points.size(), 3 );
    EXPECT_EQ( polyline.topology.lastNotLoneEdge(), EdgeId(3) ); // 2*2 = 4 half-edges in total
    EXPECT_EQ( polyline.points[v01], ( Vector2f(.5f, 0.f) ) );
}

} //namespace MR
