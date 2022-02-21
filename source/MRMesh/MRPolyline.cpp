#include "MRPolyline.h"
#include "MRPolyline2.h"
#include "MRPolylineEdgeIterator.h"
#include "MRAABBTreePolyline3.h"
#include "MRAffineXf3.h"
#include "MRVector2.h"
#include "MRGTest.h"
#include "MRTimer.h"
#include "MRMesh.h"
#include "MRComputeBoundingBox.h"
#include "MRPch/MRTBB.h"

namespace MR
{

Polyline::Polyline( const Contours2f& contours )
{
    MR_TIMER
    topology.buildFromContours( contours, 
        [&points = this->points]( size_t sz )
        {
            points.reserve( sz );
        },
        [&points = this->points]( const Vector2f & p )
        {
            points.emplace_back( p.x, p.y, 0.0f );
            return points.backId();
        } 
    );
}

EdgeId Polyline::addFromPoints( const Vector3f * vs, size_t num, bool closed )
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

    return topology.makePolyline( newVerts.data(), numSegmEnds );
}

EdgeId Polyline::addFromPoints( const Vector3f * vs, size_t num )
{
    if ( !vs || num < 2 )
    {
        assert( false );
        return {};
    }
    const bool closed = vs[0] == vs[num-1];
    return addFromPoints( vs, num - ( closed ? 1 : 0 ), closed );
}

float Polyline::totalLength() const
{
    MR_TIMER
    double sum = 0;
    for ( auto ue : undirectedEdges( topology ) )
        sum += edgeLength( ue );

    return (float)sum;
}

Box3f Polyline::getBoundingBox() const
{
    return getAABBTree().getBoundingBox();
}

Box3f Polyline::computeBoundingBox( const AffineXf3f * toWorld ) const
{
    return MR::computeBoundingBox( points, topology.getValidVerts(), toWorld );
}

Contours2f Polyline::contours() const
{
    MR_TIMER
    return topology.convertToContours<Vector2f>( 
        [&points = this->points]( VertId v )
        {
            return Vector2f{ points[v].x, points[v].y };
        } 
    );
}

Polyline2 Polyline::toPolyline2() const
{
    Polyline2 res;
    res.topology = topology;
    res.points.reserve( points.size() );
    for ( size_t i = 0; i < points.size(); i++ )
    {
        res.points.push_back( { points[VertId( i )].x, points[VertId( i )].y } );
    }
    return res;
}

EdgeId Polyline::addFromEdgePath( const Mesh& mesh, const EdgePath& path )
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
        points[newV] = mesh.orgPnt( path[i] );
    }
    if ( !closed )
    {
        newVerts.back() = VertId( shift + path.size() );
        points.back() = mesh.destPnt( path.back() );
    }
    else
    {
        newVerts.back() = newVerts.front();
    }
    return topology.makePolyline( newVerts.data(), newVerts.size() );
}

EdgeId Polyline::addFromSurfacePath( const Mesh& mesh, const SurfacePath& path )
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
        points[newV] = mesh.edgePoint( path[i] );
    }
    if ( !closed )
    {
        newVerts.back() = VertId( shift + path.size() - 1 );
        points.back() = mesh.edgePoint( path.back() );
    }
    else
    {
        newVerts.back() = newVerts.front();
    }
    return topology.makePolyline( newVerts.data(), newVerts.size() );
}

void Polyline::transform( const AffineXf3f & xf )
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

const AABBTreePolyline3& Polyline::getAABBTree() const
{
    return AABBTreeOwner_.getOrCreate( [this]{ return AABBTreePolyline3( *this ); } );
}

TEST( Polyline, Contour )
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

    Polyline pl( conts );
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

} //namespace MR
