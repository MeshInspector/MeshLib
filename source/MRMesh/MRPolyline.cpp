#include "MRPolyline.h"
#include "MRPolylineEdgeIterator.h"
#include "MRAABBTreePolyline.h"
#include "MRAffineXf2.h"
#include "MRAffineXf3.h"
#include "MRVector2.h"
#include "MRTimer.h"
#include "MRMesh.h"
#include "MRComputeBoundingBox.h"
#include "MREdgePaths.h"
#include "MRPch/MRTBB.h"

namespace MR
{

template<typename V>
Polyline<V>::Polyline( const Contour<V>& contours )
{
    addFromPoints( contours.data(), contours.size() );
}

template<typename V>
Polyline<V>::Polyline( const Contours<V>& contours )
{
    MR_TIMER;
    topology.buildFromContours( contours,
        [&points = this->points]( size_t sz )
        {
            points.reserve( sz );
        },
        [&points = this->points]( const V & p )
        {
            points.push_back( p );
            return points.backId();
        }
    );
}

template<typename V>
Polyline<V>::Polyline( const std::vector<VertId> & comp2firstVert, Vector<V, VertId> ps )
{
    MR_TIMER;
    topology.buildOpenLines( comp2firstVert );
    points = std::move( ps );
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
    const bool closed = num > 2 && vs[0] == vs[num-1];
    return addFromPoints( vs, num - ( closed ? 1 : 0 ), closed );
}

template<typename V>
void MR::Polyline<V>::addPart( const Polyline<V>& from, VertMap * outVmap, WholeEdgeMap * outEmap )
{
    MR_TIMER;

    VertMap vmap;
    VertMap* vmapPtr = outVmap ? outVmap : &vmap;
    topology.addPart( from.topology, vmapPtr, outEmap );
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
void Polyline<V>::pack( VertMap * outVmap, WholeEdgeMap * outEmap )
{
    MR_TIMER;

    Polyline<V> packed;
    packed.points.reserve( topology.numValidVerts() );
    packed.topology.vertReserve( topology.numValidVerts() );
    packed.topology.edgeReserve( 2 * topology.computeNotLoneUndirectedEdges() );
    packed.addPart( *this, outVmap, outEmap );
    *this = std::move( packed );
}

template<typename V>
EdgePoint Polyline<V>::toEdgePoint( EdgeId e, const V & p ) const
{
    const auto & po = points[ topology.org( e ) ];
    const auto & pd = points[ topology.dest( e ) ];
    const auto dt = dot( p - po , pd - po );
    const auto edgeLenSq = ( pd - po ).lengthSq();
    if ( dt <= 0 || edgeLenSq <= 0 )
        return { e, 0 };
    if ( dt >= edgeLenSq )
        return { e, 1 };
    return { e, dt / edgeLenSq };
}

template<typename V>
Vector3f MR::Polyline<V>::loopDirArea( EdgeId e0 ) const
{
    Vector3f area;
    auto e = e0;
    for ( ;; )
    {
        area += cross( Vector3f( orgPnt( e ) ), Vector3f( destPnt( e ) ) );
        e = topology.next( e.sym() );
        if ( e == e0 )
            return area;
        else if ( e == e0.sym() )
            return Vector3f( 0.0f, 0.0f, FLT_MAX );
    }
}

template<typename V>
float Polyline<V>::totalLength() const
{
    MR_TIMER;
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
V Polyline<V>::findCenterFromPoints() const
{
    MR_TIMER;
    if ( topology.numValidVerts() <= 0 )
    {
        assert( false );
        return {};
    }
    auto sumPos = parallel_deterministic_reduce( tbb::blocked_range( 0_v, VertId{ topology.vertSize() }, 1024 ), V{},
    [&] ( const auto & range, V curr )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
            if ( topology.hasVert( v ) )
                curr += points[v];
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );
    return sumPos / (float)topology.numValidVerts();
}

template<typename V>
Box<V> Polyline<V>::computeBoundingBox( const AffineXf<V> * toWorld ) const
{
    return MR::computeBoundingBox( points, topology.getValidVerts(), toWorld );
}

template<typename V>
Contours<V> Polyline<V>::contours( std::vector<std::vector<VertId>>* vertMap ) const
{
    MR_TIMER;
    return topology.convertToContours<V>(
        [&points = this->points] ( VertId v )
        {
            return points[v];
        }, vertMap
    );
}

template<typename V>
EdgeId Polyline<V>::addFromEdgePath( const Mesh& mesh, const EdgePath& path )
{
    assert( isEdgePath( mesh.topology, path ) );
    if ( path.empty() )
    {
        assert( false );
        return {};
    }

    auto v0 = topology.addVertId();
    points.autoResizeSet( v0, V{ mesh.orgPnt( path.front() ) } );
    assert( points.size() == topology.vertSize() );

    PolylineMaker maker( topology );
    const auto e0 = maker.start( v0 );
    for ( int i = 1; i < path.size(); ++i )
    {
        auto v = topology.addVertId();
        points.push_back( V{ mesh.orgPnt( path[i] ) } );
        maker.proceed( v );
    }

    bool closed = mesh.topology.org( path.front() ) == mesh.topology.dest( path.back() );
    if ( closed )
    {
        maker.close();
    }
    else
    {
        auto v = topology.addVertId();
        points.push_back( V{ mesh.destPnt( path.back() ) } );
        maker.finishOpen( v );
    }

    invalidateCaches();
    return e0;
}

template<typename V>
EdgeId Polyline<V>::addFromGeneralSurfacePath( const Mesh& mesh, const MeshTriPoint & start, const SurfacePath& path, const MeshTriPoint & end )
{
    if ( ( !start && path.empty() ) || ( !end && path.empty() ) )
    {
        assert( !start && !end );
        return {};
    }

    auto v0 = topology.addVertId();
    points.autoResizeSet( v0, V{ start ? mesh.triPoint( start ) : mesh.edgePoint( path.front() ) } );
    assert( points.size() == topology.vertSize() );

    PolylineMaker maker( topology );
    const auto e0 = maker.start( v0 );

    const bool closed = ( start && start == end ) || ( !start && path.size() > 1 && path.front() == path.back() );
    const int inc = end || closed ? 0 : 1;
    for ( int i = start ? 0 : 1; i + inc < path.size(); ++i )
    {
        auto v = topology.addVertId();
        points.push_back( V{ mesh.edgePoint( path[i] ) } );
        maker.proceed( v );
    }

    if ( closed )
    {
        maker.close();
    }
    else
    {
        auto v = topology.addVertId();
        points.push_back( V{ end ? mesh.triPoint( end ) : mesh.edgePoint( path.back() ) } );
        maker.finishOpen( v );
    }

    invalidateCaches();
    return e0;
}

template<typename V>
void Polyline<V>::transform( const AffineXf<V> & xf )
{
    MR_TIMER;
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

} //namespace MR
