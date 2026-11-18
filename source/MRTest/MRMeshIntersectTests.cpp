#include <gtest/gtest.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshIntersect.h>
#include <MRMesh/MRLine3.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MREdgeIterator.h>
#include <MRMesh/MRAABBTree.h>
#include <MRMesh/MRTriMesh.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRPlane3.h>
#include <MRMesh/MRIntersectionPrecomputes.h>
#include <MRMesh/MRTriangleIntersection.h>
#include <algorithm>
#include <cmath>

namespace MR
{

TEST(MRMesh, MeshIntersect) 
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );

    std::vector<MeshIntersectionResult> allFound;
    auto callback = [&allFound]( const MeshIntersectionResult & found ) -> bool
    {
        allFound.push_back( found );
        return true;
    };

    Vector3f d{ 1, 2, 3 };
    rayMeshIntersectAll( sphere, { 2.0f * d, -d.normalized() }, callback );
    ASSERT_EQ( allFound.size(), 2 );
    for ( const auto & found : allFound )
    {
        ASSERT_NEAR( found.proj.point.length(), 1.0f, 0.05f ); //our sphere is very approximate
    }

    const auto isect1 = rayMeshIntersect( sphere, { Vector3f{ +0.1f, 0.f, 0.f }, Vector3f::plusX() }, -FLT_MAX, +FLT_MAX );
    EXPECT_TRUE( isect1 );
    EXPECT_NEAR( isect1.distanceAlongLine, 0.9f, 0.05f );
    EXPECT_NEAR( isect1.proj.point.x, +1.f, 0.05f );

    const auto isect2 = rayMeshIntersect( sphere, { Vector3f{ -0.1f, 0.f, 0.f }, Vector3f::plusX() }, -FLT_MAX, +FLT_MAX );
    EXPECT_TRUE( isect2 );
    EXPECT_NEAR( isect2.distanceAlongLine, -0.9f, 0.05f );
    EXPECT_NEAR( isect2.proj.point.x, -1.f, 0.05f );
}

TEST(MRMesh, MeshIntersectAllDistanceAlongLine)
{
    // non-unit direction: faces x=-0.5 and x=+0.5 are at t=0.25 and t=0.75
    Mesh cube = makeCube();
    const Vector3d p( -1, 0.1, 0.2 ), d( 2, 0, 0 );
    for ( bool useDouble : { false, true } )
    {
        std::vector<float> ts;
        auto callback = [&ts] ( const MeshIntersectionResult & found ) { ts.push_back( found.distanceAlongLine ); return true; };
        if ( useDouble )
            rayMeshIntersectAll( cube, Line3d( p, d ), callback, 0.0, 1.0 );
        else
            rayMeshIntersectAll( cube, Line3f( Vector3f( p ), Vector3f( d ) ), callback, 0.0f, 1.0f );
        std::sort( ts.begin(), ts.end() );
        ASSERT_EQ( ts.size(), 2 );
        EXPECT_NEAR( ts[0], 0.25f, 1e-6f );
        EXPECT_NEAR( ts[1], 0.75f, 1e-6f );
    }
}

TEST( MRMesh, MeshIntersectAllPrecise )
{
    auto countHits = [] ( const Mesh & mesh, const Line3d & line, double rayStart )
    {
        int numHits = 0;
        rayMeshIntersectAll( mesh, line, [&numHits] ( const MeshIntersectionResult & ) { ++numHits; return true; }, rayStart, DBL_MAX );
        return numHits;
    };

    // the face of the cube at x=0 lies in the plane of the bounding box, and the segment's end clipped by the box must not land on it
    const auto cube1 = makeCube( Vector3f::diagonal( 1 ), Vector3f::diagonal( -1 ) );
    EXPECT_EQ( countHits( cube1, Line3d( Vector3d( -2, -0.3, -0.6 ), Vector3d::plusX() ), -DBL_MAX ), 2 );

    // the ray passes the vertex in the center of the face x=-0.5 closer than the step of integer coordinates,
    // and the triangle below the vertex must not be culled
    auto cube2 = makeCube();
    for ( auto ue : undirectedEdges( cube2.topology ) )
    {
        const auto a = cube2.orgPnt( ue );
        const auto b = cube2.destPnt( ue );
        if ( a.x == -0.5f && b.x == -0.5f && ( a - b ).length() > 1.1f )
        {
            cube2.splitEdge( ue );
            break;
        }
    }
    EXPECT_EQ( countHits( cube2, Line3d( Vector3d( -1, 0, 1e-12 ), Vector3d::plusX() ), -DBL_MAX ), 2 );

    // the rays start exactly on a face, so only the precise predicates decide whether the start is inside the mesh,
    // and the rays in opposite directions must agree
    const auto cube3 = makeCube();
    for ( double x : { -0.5, 0.5 } )
    {
        const Vector3d p( x, 0.1, 0.2 );
        EXPECT_EQ( countHits( cube3, Line3d( p, Vector3d::plusX() ), 0 ) % 2, countHits( cube3, Line3d( p, Vector3d::minusX() ), 0 ) % 2 );
    }
}

TEST( MRMesh, MeshIntersectNearlyCoplanarRay )
{
    // the ray is almost in the plane of the triangle and misses it; tiny Y-coordinates must not be hidden by rounding tolerance
    const auto mesh = Mesh::fromTriangles( { Vector3f( -63, 1.65e-13f, 87.05f ), Vector3f( -62.66f, 2.82e-13f, 87.72f ), Vector3f( -47.62f, -1.88e-13f, 84.84f ) },
        { { 0_v, 1_v, 2_v } } );
    const Vector3d p( -200, 0, 87.5 );
    EXPECT_FALSE( rayMeshIntersect( mesh, Line3f( Vector3f( p ), Vector3f::plusX() ) ) );
    EXPECT_FALSE( rayMeshIntersect( mesh, Line3d( p, Vector3d::plusX() ) ) );

    // the ray across the plane hits the triangle in its centroid
    const Vector3f c = ( mesh.points[0_v] + mesh.points[1_v] + mesh.points[2_v] ) / 3.0f;
    const auto hit = rayMeshIntersect( mesh, Line3f( c + Vector3f( 0, 1, 0 ), Vector3f::minusY() ) );
    ASSERT_TRUE( hit );
    EXPECT_NEAR( hit.distanceAlongLine, 1.0f, 1e-6f );
}

// the closest intersection found with the tree is the closest one among all triangles
TEST( MRMesh, MeshIntersectAsBruteForce )
{
    const Mesh mesh = makeSphere( { .radius = 1.0f, .numMeshVertices = 2000 } );
    FaceBitSet region( mesh.topology.faceSize() );
    for ( auto f : mesh.topology.getValidFaces() )
        if ( int( f ) % 3 != 0 )
            region.set( f );

    const FaceBitSet * regions[] = { nullptr, &region };
    for ( const FaceBitSet * reg : regions )
    {
        int numHits = 0;
        for ( int i = 0; i < 2000; ++i )
        {
            // origins inside and outside of the sphere, some directions along the axes
            const Vector3f org( std::sin( i * 0.37f ) * 1.5f, std::cos( i * 0.53f ) * 1.5f, std::sin( i * 0.71f + 1 ) * 1.5f );
            Vector3f dir( std::cos( i * 1.13f ), std::sin( i * 0.97f ), std::cos( i * 0.29f + 2 ) );
            if ( i % 10 == 0 )
                dir[i % 3] = 0;
            const Line3f line( org, dir );
            const IntersectionPrecomputes<float> prec( dir );
            float bestT = FLT_MAX;
            for ( auto f : mesh.topology.getValidFaces() )
            {
                if ( reg && !reg->test( f ) )
                    continue;
                const auto [a, b, c] = mesh.topology.getTriVerts( f );
                const auto isect = rayTriangleIntersect( mesh.points[a] - org, mesh.points[b] - org, mesh.points[c] - org, prec );
                if ( isect && isect->t > 0 && isect->t < bestT )
                    bestT = isect->t;
            }
            const auto hit = rayMeshIntersect( { mesh, reg }, line );
            if ( bestT == FLT_MAX )
            {
                EXPECT_FALSE( hit );
                continue;
            }
            ++numHits;
            ASSERT_TRUE( hit );
            // equal on x64; on ARM64 the compilers contract the scalar rayTriangleIntersect into fused multiply-adds, unlike the SIMD code
            EXPECT_NEAR( hit.distanceAlongLine, bestT, 1e-5f * std::max( 1.0f, bestT ) );
            EXPECT_TRUE( !reg || reg->test( hit.proj.face ) );
            const auto anyHit = rayMeshIntersect( { mesh, reg }, line, 0.0f, FLT_MAX, nullptr, false );
            EXPECT_TRUE( anyHit );
        }
        EXPECT_GT( numHits, 200 );
    }
}

// the intersections with the triangles of a tree and their points are the same as with the mesh part of these triangles
TEST( MRMesh, MeshIntersectPointsTree )
{
    const Mesh mesh = makeUVSphere( 1.f, 64, 32 );
    FaceBitSet region( mesh.topology.faceSize() );
    for ( FaceId f : mesh.topology.getValidFaces() )
        region.set( f, f % 3 != 0 );
    const TriMesh tmesh{ mesh.topology.getTriangulation(), mesh.points };
    const AABBTree fullTree( tmesh ); // the same as mesh.getAABBTree()
    const AABBTree regionTree( tmesh, &region );
    const MeshPart mp( mesh, &region );
    const AABBTree mpTree( mp ); // the same as regionTree

    auto same = []( const MeshIntersectionResult & a, const MeshIntersectionResult & b )
    {
        return a.proj.face == b.proj.face && a.proj.point == b.proj.point && a.tp == b.tp && a.distanceAlongLine == b.distanceAlongLine;
    };
    auto all = []( auto && find )
    {
        std::vector<std::pair<MeshIntersectionResult, bool>> res;
        find( res );
        std::sort( res.begin(), res.end(), []( const auto & x, const auto & y ) { return x.first.proj.face < y.first.proj.face; } );
        return res;
    };
    auto sameAll = [&]( const auto & a, const auto & b )
    {
        if ( a.size() != b.size() )
            return false;
        for ( size_t i = 0; i < a.size(); ++i )
            if ( !same( a[i].first, b[i].first ) || a[i].second != b[i].second )
                return false;
        return true;
    };

    int numHits = 0, numAllHits = 0;
    std::vector<Vector3f> origins, dirs;
    for ( const Vector3f d : { Vector3f( 1, 0.1f, 0.2f ), Vector3f( -0.3f, 1, 0.4f ), Vector3f( 0.2f, -0.5f, -1 ) } )
    {
        for ( const Vector3f o : { Vector3f(), Vector3f( 0.1f, -0.2f, 0.05f ), Vector3f( -3, 0, 0 ) } )
        {
            origins.push_back( o );
            dirs.push_back( d );
            const Line3f lf( o, d );
            const Line3d ld( lf );
            for ( bool closest : { true, false } )
            {
                const auto ref = rayMeshIntersect( mp, lf, -FLT_MAX, FLT_MAX, nullptr, closest );
                numHits += ref.valid();
                EXPECT_TRUE( same( rayMeshIntersect( fullTree, lf, -FLT_MAX, FLT_MAX, nullptr, closest, {}, &region ), ref ) );
                EXPECT_TRUE( same( rayMeshIntersect( fullTree, ld, -DBL_MAX, DBL_MAX, nullptr, closest, {}, &region ),
                    rayMeshIntersect( mp, ld, -DBL_MAX, DBL_MAX, nullptr, closest ) ) );
                EXPECT_TRUE( same( rayMeshIntersect( regionTree, lf, -FLT_MAX, FLT_MAX, nullptr, closest ),
                    rayMeshIntersect( mpTree, lf, -FLT_MAX, FLT_MAX, nullptr, closest, {}, &region ) ) );
            }

            auto collect = []( auto & res ) { return [&res]( const MeshIntersectionResult & r ) { res.push_back( { r, false } ); return true; }; };
            auto collectSide = []( auto & res ) { return [&res]( const MeshIntersectionResult & r, bool front ) { res.push_back( { r, front } ); return true; }; };
            const auto refF = all( [&]( auto & res ) { rayMeshIntersectAll( mp, lf, collect( res ), -FLT_MAX, FLT_MAX ); } );
            const auto refD = all( [&]( auto & res ) { rayMeshIntersectAll( mp, ld, collect( res ), -DBL_MAX, DBL_MAX ); } );
            const auto refS = all( [&]( auto & res ) { rayMeshIntersectAll( mp, ld, MeshIntersectionWithSideCallback( collectSide( res ) ), -DBL_MAX, DBL_MAX ); } );
            numAllHits += int( refS.size() );
            EXPECT_TRUE( sameAll( all( [&]( auto & res ) { rayMeshIntersectAll( fullTree, lf, collect( res ), -FLT_MAX, FLT_MAX, nullptr, &region ); } ), refF ) );
            EXPECT_TRUE( sameAll( all( [&]( auto & res ) { rayMeshIntersectAll( fullTree, ld, collect( res ), -DBL_MAX, DBL_MAX, nullptr, &region ); } ), refD ) );
            EXPECT_TRUE( sameAll( all( [&]( auto & res ) { rayMeshIntersectAll( fullTree, ld,
                MeshIntersectionWithSideCallback( collectSide( res ) ), -DBL_MAX, DBL_MAX, nullptr, &region ); } ), refS ) );
            EXPECT_TRUE( sameAll( all( [&]( auto & res ) { rayMeshIntersectAll( regionTree, ld,
                MeshIntersectionWithSideCallback( collectSide( res ) ), -DBL_MAX, DBL_MAX ); } ), refS ) );
        }
    }
    EXPECT_GT( numHits, 0 );
    EXPECT_GT( numAllHits, 0 );

    std::vector<FaceId> facesA, facesB;
    std::vector<float> distA, distB;
    std::vector<TriPointf> baryA, baryB;
    multiRayMeshIntersect( mp, origins, dirs, { .rayDistances = &distA, .isectFaces = &facesA, .isectBary = &baryA } );
    multiRayMeshIntersect( fullTree, origins, dirs, { .rayDistances = &distB, .isectFaces = &facesB, .isectBary = &baryB },
        0.0f, FLT_MAX, true, {}, &region );
    EXPECT_EQ( facesA, facesB );
    for ( size_t i = 0; i < origins.size(); ++i )
        EXPECT_TRUE( ( std::isnan( distA[i] ) && std::isnan( distB[i] ) ) || ( distA[i] == distB[i] && baryA[i] == baryB[i] ) );

    for ( float z : { -0.5f, 0.f, 0.7f } )
    {
        FaceBitSet fsA( mesh.topology.faceSize() ), fsB;
        VertBitSet vsA( mesh.topology.vertSize() ), vsB;
        xyPlaneMeshIntersect( mp, z, &fsA, nullptr, &vsA );
        xyPlaneMeshIntersect( fullTree, z, &fsB, &vsB, nullptr, &region );
        fsB.resize( fsA.size() );
        vsB.resize( vsA.size() );
        EXPECT_TRUE( fsA.any() && fsA == fsB && vsA == vsB );

        const auto plane = Plane3f::fromDirAndPt( Vector3f( 0.3f, -0.2f, 1 ).normalized(), Vector3f( 0, 0, z ) );
        std::vector<FaceId> vecA, vecB;
        planeMeshIntersect( mp, plane, nullptr, nullptr, nullptr, &vecA );
        planeMeshIntersect( regionTree, plane, nullptr, nullptr, &vecB );
        std::sort( vecA.begin(), vecA.end() );
        std::sort( vecB.begin(), vecB.end() );
        EXPECT_TRUE( !vecA.empty() && vecA == vecB );
    }
}

} //namespace MR
