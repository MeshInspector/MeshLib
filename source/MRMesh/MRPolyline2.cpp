#include "MRPolyline2.h"
#include "MRPolyline.h"
#include "MRPolylineEdgeIterator.h"
#include "MRAABBTreePolyline2.h"
#include "MRAffineXf2.h"
#include "MRVector2.h"
#include "MRGTest.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

Polyline2::Polyline2( const Contours2f& contours )
{
    MR_TIMER
    topology.buildFromContours( contours, 
        [&points = this->points]( size_t sz )
        {
            points.reserve( sz );
        },
        [&points = this->points]( const Vector2f & p )
        {
            points.emplace_back( p.x, p.y );
            return points.backId();
        } 
    );
}

float Polyline2::totalLength() const
{
    MR_TIMER
    double sum = 0;
    for ( auto ue : undirectedEdges( topology ) )
        sum += edgeLength( ue );

    return (float)sum;
}

Box2f Polyline2::getBoundingBox() const
{
    return getAABBTree().getBoundingBox();
}

Contours2f Polyline2::contours() const
{
    MR_TIMER
    return topology.convertToContours<Vector2f>( 
        [&points = this->points]( VertId v )
        {
            return Vector2f{ points[v].x, points[v].y };
        } 
    );
}

Polyline Polyline2::toPolyline3() const
{
    Polyline res;
    res.topology = topology;
    res.points.reserve( points.size() );
    for ( size_t i = 0; i < points.size(); i++ )
    {
        res.points.push_back( { points[VertId( i )].x, points[VertId( i )].y, 0.f } );
    }
    return res;
}

void Polyline2::transform( const AffineXf2f & xf )
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

const AABBTreePolyline2& Polyline2::getAABBTree() const
{
    return AABBTreeOwner_.getOrCreate( [this]{ return AABBTreePolyline2( *this ); } );
}

TEST( Polyline2, Contour )
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

} //namespace MR
