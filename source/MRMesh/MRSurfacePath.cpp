#include "MRSurfacePath.h"
#include "MRSurfaceDistance.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRRingIterator.h"
#include "MRPlanarPath.h"
#include "MRTimer.h"
#include "MRGTest.h"
#include "MRRegionBoundary.h"
#include "MRMeshLoad.h"
#include "MRBitSetParallelFor.h"
#include "MRMeshBuilder.h"
#include "MREdgePaths.h"

namespace MR
{

// consider triangle 0bc, where a linear scalar field is defined in all vertices: v(0) = 0, v(b) = vb, v(c) = vc;
// computes field gradient in the triangle
static Vector3d computeGradient( const Vector3d & b, const Vector3d & c, double vb, double vc )
{
    const auto bb = dot( b, b );
    const auto bc = dot( b, c );
    const auto cc = dot( c, c );
    const auto det = bb * cc - bc * bc;
    if ( det <= 0 )
    {
        // degenerate triangle
        return {};
    }
    const auto kb = ( 1 / det ) * ( cc * vb - bc * vc );
    const auto kc = ( 1 / det ) * (-bc * vb + bb * vc );
    return kb * b + kc * c;
}

static Vector3f computeGradient( const Vector3f & b, const Vector3f & c, float vb, float vc )
{
    return Vector3f{ computeGradient( Vector3d( b ), Vector3d( c ), double( vb ), double( vc ) ) };
}

// consider triangle 0bc, where gradient is given;
// computes the intersection of the ray (org=0, dir=-grad) with the open segment (b,c)
static std::optional<float> computeExitPos( const Vector3f & b, const Vector3f & c, const Vector3f & grad )
{
    const auto gradSq = grad.lengthSq();
    if ( gradSq <= 0 )
        return {};
    const auto d = c - b;
    // gort is a vector in the triangle plane orthogonal to grad
    const auto gort = d - ( dot( d, grad ) / gradSq ) * grad;
    const auto god = dot( gort, d );
    if ( god <= 0 )
        return {};
    const auto gob = -dot( gort, b );
    if ( gob <= 0 || gob >= god )
        return {};
    const auto a = gob / god;
    assert( a < FLT_MAX );
    const auto ip = a * c + ( 1 - a ) * b;
    if ( dot( grad, ip ) >= 0 )
        return {}; // (b,c) is intersected in the direction +grad
    return a;
}

class SurfacePathBuilder
{
public:
    SurfacePathBuilder( const Mesh & mesh, const Vector<float,VertId> & field );

    // finds previous path point before given vertex, which can be located on any first ring boundary
    std::optional<MeshEdgePoint> findPrevPoint( VertId v ) const;
    // finds previous path point before given edge location
    std::optional<MeshEdgePoint> findPrevPoint( const MeshEdgePoint & ep ) const;
    // finds previous path point before given triangle location
    std::optional<MeshEdgePoint> findPrevPoint( const MeshTriPoint & tp ) const;

private:
    const Mesh & mesh_;
    const Vector<float,VertId> & field_;
};

SurfacePathBuilder::SurfacePathBuilder( const Mesh & mesh, const Vector<float,VertId> & field )
    : mesh_( mesh )
    , field_( field )
{
}

std::optional<MeshEdgePoint> SurfacePathBuilder::findPrevPoint( VertId v ) const
{
    std::optional<MeshEdgePoint> res;
    float maxGradSq = 0;
    const auto vv = field_[v];
    const auto pv = mesh_.points[v];
    for ( EdgeId e : orgRing( mesh_.topology, v ) )
    {
        const auto d = mesh_.topology.dest( e );
        const auto pd = mesh_.points[d] - pv;
        const auto pdSq = pd.lengthSq();
        const auto vd = field_[d] - vv;
        if ( vd < 0 )
        {
            if ( pdSq == 0 && maxGradSq == 0 && !res ) // degenerate edge
                res = MeshEdgePoint{ e.sym(), 0 };
            else if ( pdSq > 0 )
            {
                auto edgeGradSq = sqr( vd ) / pdSq;
                if ( edgeGradSq > maxGradSq )
                {
                    maxGradSq = edgeGradSq;
                    res = MeshEdgePoint{ e.sym(), 0 };
                }
            }
        }
        if ( mesh_.topology.left( e ) )
        {
            const auto eBd = mesh_.topology.prev( e.sym() );
            const auto x = mesh_.topology.dest( eBd );
            const auto px = mesh_.points[x] - pv;
            if ( auto fx = field_[x]; fx < FLT_MAX )
            {
                const auto vx = fx - vv;
                const auto triGrad = computeGradient( pd, px, vd, vx );
                const auto triGradSq = triGrad.lengthSq();
                if ( triGradSq > maxGradSq )
                {
                    if ( auto a = computeExitPos( pd, px, triGrad ) )
                    {
                        maxGradSq = triGradSq;
                        res = MeshEdgePoint{ eBd, *a };
                    }
                }
            }
        }
    }
    return res;
}

std::optional<MeshEdgePoint> SurfacePathBuilder::findPrevPoint( const MeshEdgePoint & ep ) const
{
    if ( auto v = ep.inVertex( mesh_.topology ) )
        return findPrevPoint( v );

    // point is not in vertex
    std::optional<MeshEdgePoint> result;
    float maxGradSq = 0;
    const auto p = mesh_.edgePoint( ep );

    const auto o = mesh_.topology.org( ep.e );
    const auto d = mesh_.topology.dest( ep.e );
    const auto fo = field_[o];
    const auto fd = field_[d];
    const auto v = ( 1 - ep.a ) * fo + ep.a * fd;
    const auto vo = fo - v;
    const auto vd = fd - v;
    const auto po = mesh_.points[o] - p;
    const auto pd = mesh_.points[d] - p;

    // stores candidate in the result if it has smaller value than initial point
    auto updateRes = [&result, v]( const MeshEdgePoint & candidateEdgePoint, float edgeOrgValue, float edgeDestValue )
    {
        const float candidateValue = edgeOrgValue * ( 1 - candidateEdgePoint.a ) + edgeDestValue * candidateEdgePoint.a;
        if ( v <= candidateValue )
            return false;
        result = candidateEdgePoint;
        return true;
    };

    if ( fo < fd )
    {
        const auto poSq = po.lengthSq();
        if ( poSq >= 0 ) // not strict to handle cases with `inVertex` fail but coordinates same as vertex
        {
            result = MeshEdgePoint{ ep.e, 0 };
            maxGradSq = poSq == 0 ? 0.0f : ( sqr( vo ) / poSq );
        }
    }
    else if ( fd < fo )
    {
        const auto pdSq = pd.lengthSq();
        if ( pdSq >= 0 ) // not strict to handle cases with `inVertex` fail but coordinates same as vertex
        {
            result = MeshEdgePoint{ ep.e.sym(), 0 };
            maxGradSq = pdSq == 0 ? 0.0f : ( sqr( vd ) / pdSq );
        }
    }

    if ( mesh_.topology.left( ep.e ) )
    {
        const auto el = mesh_.topology.next( ep.e );
        const auto l = mesh_.topology.dest( el );
        const auto fl = field_[l];
        if ( fl < FLT_MAX )
        {
            const auto vl = fl - v;
            const auto pl = mesh_.points[l] - p;
            const auto plSq = pl.lengthSq();
            if ( vl < 0 && plSq > 0 )
            {
                auto edgeGradSq = sqr( vl ) / plSq;
                if ( edgeGradSq > maxGradSq )
                {
                    result = MeshEdgePoint{ el.sym(), 0 };
                    maxGradSq = edgeGradSq;
                }
            }

            const auto triGrad = computeGradient( pd - po, pl - po, vd - vo, vl - vo );
            const auto triGradSq = triGrad.lengthSq();
            if ( triGradSq > maxGradSq )
            {
                if ( auto a0 = computeExitPos( pd, pl, triGrad ) )
                {
                    if ( updateRes( MeshEdgePoint{ mesh_.topology.prev( ep.e.sym() ), *a0 }, fd, fl ) )
                        maxGradSq = triGradSq;
                }
                else if ( auto a1 = computeExitPos( pl, po, triGrad ) )
                {
                    if ( updateRes( MeshEdgePoint{ el.sym(), *a1 }, fl, fo ) )
                        maxGradSq = triGradSq;
                }
            }
        }
    }

    if ( mesh_.topology.right( ep.e ) )
    {
        const auto er = mesh_.topology.prev( ep.e );
        const auto r = mesh_.topology.dest( er );
        const auto fr = field_[r];
        if ( fr < FLT_MAX )
        {
            const auto vr = fr - v;
            const auto pr = mesh_.points[r] - p;
            const auto prSq = pr.lengthSq();
            if ( vr < 0 && prSq > 0 )
            {
                auto edgeGradSq = sqr( vr ) / prSq;
                if ( edgeGradSq > maxGradSq )
                {
                    result = MeshEdgePoint{ er.sym(), 0 };
                    maxGradSq = edgeGradSq;
                }
            }

            const auto triGrad = computeGradient( pr - po, pd - po, vr - vo, vd - vo );
            const auto triGradSq = triGrad.lengthSq();
            if ( triGradSq > maxGradSq )
            {
                if ( auto a0 = computeExitPos( pr, pd, triGrad ) )
                {
                    if ( updateRes( MeshEdgePoint{ mesh_.topology.next( ep.e.sym() ).sym(), *a0 }, fr, fd ) )
                        maxGradSq = triGradSq;
                }
                else if ( auto a1 = computeExitPos( po, pr, triGrad ) )
                {
                    if ( updateRes( MeshEdgePoint{ er, *a1 }, fo, fr ) )
                        maxGradSq = triGradSq;
                }
            }
        }
    }

    return result;
}

std::optional<MeshEdgePoint> SurfacePathBuilder::findPrevPoint( const MeshTriPoint & tp ) const
{
    if ( auto ep = tp.onEdge( mesh_.topology ) )
        return findPrevPoint( *ep );

    // point is not on edge
    std::optional<MeshEdgePoint> res;
    float maxGradSq = -1;
    const auto p = mesh_.triPoint( tp );

    VertId v[3];
    mesh_.topology.getLeftTriVerts( tp.e, v );

    Vector3f pv[3];
    float vv[3];
    EdgeId e[3];
    auto ei = tp.e;
    for ( int i = 0; i < 3; ++i )
    {
        pv[i] = mesh_.points[v[i]] - p;
        vv[i] = field_[v[i]];
        e[i] = ei;
        ei = mesh_.topology.prev( ei.sym() );
    }
    const auto f = tp.bary.interpolate( vv[0], vv[1], vv[2] );

    for ( int i = 0; i < 3; ++i )
    {
        vv[i] -= f;
        if ( vv[i] < 0 )
        {
            const auto pvSq = pv[i].lengthSq();
            // if input point is close to a triangle vertex then pvSq can be zero
            auto edgeGradSq = pvSq > 0 ? sqr( vv[i] ) / pvSq : 0;
            if ( edgeGradSq > maxGradSq )
            {
                maxGradSq = edgeGradSq;
                res = MeshEdgePoint{ e[i], 0 };
            }
        }
    }

    const auto triGrad = computeGradient( pv[1] - pv[0], pv[2] - pv[0], vv[1] - vv[0], vv[2] - vv[0] );
    const auto triGradSq = triGrad.lengthSq();
    if ( triGradSq > maxGradSq )
    {
        for ( int i = 0; i < 3; ++i )
        {
            if ( auto a = computeExitPos( pv[i], pv[( i + 1 ) % 3], triGrad ) )
            {
                maxGradSq = triGradSq;
                res = MeshEdgePoint{ e[i], *a };
            }
        }
    }

    return res;
}

tl::expected<SurfacePath, PathError> computeGeodesicPathApprox( const Mesh & mesh,
    const MeshTriPoint & start, const MeshTriPoint & end, GeodesicPathApprox atype )
{
    MR_TIMER;
    if ( atype == GeodesicPathApprox::FastMarching )
        return computeFastMarchingPath( mesh, start, end );

    SurfacePath res;
    if ( !fromSameTriangle( mesh.topology, MeshTriPoint{ start }, MeshTriPoint{ end } ) )
    {
        VertId v1, v2;
        EdgePath edgePath = ( atype == GeodesicPathApprox::DijkstraBiDir ) ?
            buildShortestPathBiDir( mesh, start, end, &v1, &v2 ) :
            buildShortestPathAStar( mesh, start, end, &v1, &v2 );
        if ( !v1 || !v2 )
            return tl::make_unexpected( PathError::StartEndNotConnected );

        // remove last segment from the path if end-point and the origin of last segment belong to one triangle
        while( !edgePath.empty()
            && fromSameTriangle( mesh.topology, MeshTriPoint{ end }, MeshTriPoint{ MeshEdgePoint{ edgePath.back(), 0 } } ) )
        {
            v2 = mesh.topology.org( edgePath.back() );
            edgePath.pop_back();
        }

        // remove first segment from the path if start-point and the destination of first segment belong to one triangle
        while( !edgePath.empty()
            && fromSameTriangle( mesh.topology, MeshTriPoint{ start }, MeshTriPoint{ MeshEdgePoint{ edgePath.front(), 1 } } ) )
        {
            v1 = mesh.topology.dest( edgePath.front() );
            edgePath.erase( edgePath.begin() );
        }

        if ( edgePath.empty() )
        {
            assert ( v1 == v2 );
            res = { MeshEdgePoint( mesh.topology.edgeWithOrg( v1 ), 0.0f ) };
        }
        else
        {
            res.reserve( edgePath.size() + 1 );
            for ( EdgeId e : edgePath )
                res.emplace_back( e, 0.0f );
            res.emplace_back( edgePath.back(), 1.0f );
        }
    }
    return res;
}

tl::expected<std::vector<MeshEdgePoint>, PathError> computeFastMarchingPath( const MeshPart & mp,
    const MeshTriPoint & start, const MeshTriPoint & end,
    const VertBitSet* vertRegion, Vector<float, VertId> * outSurfaceDistances )
{
    MR_TIMER;
    std::vector<MeshEdgePoint> res;
    auto s = start;
    auto e = end;
    if ( fromSameTriangle( mp.mesh.topology, s, e ) )
        return res; // path does not cross any edges

    // the region can be specified by faces or by vertices, but not in both ways at the same time
    assert( !mp.region || !vertRegion );

    VertBitSet myVertRegion;
    if ( mp.region )
    {
        myVertRegion = getIncidentVerts( mp.mesh.topology, *mp.region );
        vertRegion = &myVertRegion;
    }

    // build distances from end to start, so to get correct path in reverse order
    bool connected = false;
    auto distances = computeSurfaceDistances( mp.mesh, end, start, vertRegion, &connected );
    if ( !connected )
        return tl::make_unexpected( PathError::StartEndNotConnected );

    SurfacePathBuilder b( mp.mesh, distances );
    auto curr = b.findPrevPoint( start );
    assert( curr ); // it should be if start and end are not from the same triangle
    while ( curr )
    {
        res.push_back( *curr );
        MeshTriPoint c( *curr );
        if ( fromSameTriangle( mp.mesh.topology, e, c ) )
            break; // reached triangle with end point
        if ( res.size() > mp.mesh.topology.numValidFaces() )
            return tl::make_unexpected( PathError::InternalError ); // normal path cannot visit any triangle more than once
        curr = b.findPrevPoint( *curr );
    }
    assert( !res.empty() );

    if ( outSurfaceDistances )
        *outSurfaceDistances = std::move( distances );
    return res;
}

tl::expected<std::vector<MeshEdgePoint>, PathError> computeSurfacePath( const MeshPart & mp,
    const MeshTriPoint & start, const MeshTriPoint & end, int maxGeodesicIters,
    const VertBitSet* vertRegion, Vector<float, VertId> * outSurfaceDistances )
{
    MR_TIMER;
    auto res = computeFastMarchingPath( mp, start, end, vertRegion, outSurfaceDistances );
    if ( res.has_value() && !res.value().empty() )
        reducePath( mp.mesh, start, res.value(), end, maxGeodesicIters );
    return res;
}

tl::expected<SurfacePath, PathError> computeGeodesicPath( const Mesh & mesh,
    const MeshTriPoint & start, const MeshTriPoint & end, GeodesicPathApprox atype,
    int maxGeodesicIters )
{
    MR_TIMER;
    auto res = computeGeodesicPathApprox( mesh, start, end, atype );
    if ( res.has_value() && !res.value().empty() )
        reducePath( mesh, start, res.value(), end, maxGeodesicIters );
    return res;
}

HashMap<VertId, VertId> computeClosestSurfacePathTargets( const Mesh & mesh,
    const VertBitSet & starts, const VertBitSet & ends, 
    const VertBitSet * vertRegion, Vector<float, VertId> * outSurfaceDistances )
{
    MR_TIMER;
    auto distances = computeSurfaceDistances( mesh, ends, starts, FLT_MAX, vertRegion );

    HashMap<VertId, VertId> res;
    res.reserve( starts.count() );
    // create all keys in res before parallel region
    for ( auto v : starts )
        res[v];

    BitSetParallelFor( starts, [&]( VertId v )
    {
        SurfacePathBuilder b( mesh, distances );
        auto last = b.findPrevPoint( v );
        // if ( !last ) then v is not reachable from (ends) or it is contained in (ends)
        int steps = 0;
        while ( last )
        {
            if ( ++steps > mesh.topology.numValidFaces() )
            {
                // internal error
                assert( false );
                last.reset();
                break;
            }
            if ( auto next = b.findPrevPoint( *last ) )
                last = next;
            else
                break;
        }
        if ( last )
            res[v] = last->getClosestVertex( mesh.topology );
    } );

    if ( outSurfaceDistances )
        *outSurfaceDistances = std::move( distances );
    return res;
}

float surfacePathLength( const Mesh& mesh, const SurfacePath& surfacePath )
{
    if ( surfacePath.empty() )
        return 0.0f;
    float sum = 0.0f;
    auto prevPoint = mesh.edgePoint( surfacePath[0] );
    for ( int i = 1; i < surfacePath.size(); ++i )
    {
        auto curPoint = mesh.edgePoint( surfacePath[i] );
        sum += ( curPoint - prevPoint ).length();
        prevPoint = curPoint;
    }
    return sum;
}

TEST(MRMesh, SurfacePath) 
{
    Vector3f g;

    g = computeGradient( Vector3f{ 1, 0, 0 }, Vector3f{ 0.5f, 1, 0 }, 0, 1 );
    EXPECT_NEAR( ( g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    g = computeGradient( Vector3f{ 1, 0, 0 }, Vector3f{ 0.1f, 1, 0 }, 0, 1 );
    EXPECT_NEAR( ( g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    g = computeGradient( Vector3f{ 1, 0, 0 }, Vector3f{ 0.9f, 1, 0 }, 0, 1 );
    EXPECT_NEAR( ( g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    std::optional<float> e;
    g = computeGradient( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, 1, 1 );
    EXPECT_NEAR( ( g - Vector3f{ 1, 1, 0 } ).length(), 0, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, g );
    EXPECT_FALSE( e.has_value() );
    e = computeExitPos ( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, -g );
    EXPECT_NEAR( *e, 0.5f, 1e-5f );

    g = computeGradient( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, -1, -1 );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, g );
    EXPECT_NEAR( *e, 0.5f, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, -g );
    EXPECT_FALSE( e.has_value() );

    g = computeGradient( Vector3f{ 1, -0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -1, -1 );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, -0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, g );
    EXPECT_NEAR( *e, 0.1f, 1e-5f );

    g = computeGradient( Vector3f{ 1, -0.9f, 0 }, Vector3f{ 1, 0.1f, 0 }, -1, -1 );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, -0.9f, 0 }, Vector3f{ 1, 0.1f, 0 }, g );
    EXPECT_NEAR( *e, 0.9f, 1e-5f );

    g = computeGradient( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -1, -1 );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, g );
    EXPECT_FALSE( e.has_value() );
    e = computeExitPos ( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -g );
    EXPECT_FALSE( e.has_value() );
}

TEST( MRMesh, SurfacePathTargets )
{
    Triangulation t{
        { 0_v, 1_v, 2_v }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );

    mesh.points.emplace_back( 0.f, 0.f, 0.f ); // 0_v
    mesh.points.emplace_back( 1.f, 0.f, 0.f ); // 1_v
    mesh.points.emplace_back( 0.f, 1.f, 0.f ); // 2_v

    VertBitSet starts(3);
    starts.set( 1_v );
    starts.set( 2_v );

    VertBitSet ends(3);
    ends.set( 0_v );

    const auto map = computeClosestSurfacePathTargets( mesh, starts, ends );
    EXPECT_EQ( map.size(), starts.count() );
    for ( const auto & [start, end] : map )
    {
        EXPECT_TRUE( starts.test( start ) );
        EXPECT_TRUE( ends.test( end ) );
    }
}

} //namespace MR
