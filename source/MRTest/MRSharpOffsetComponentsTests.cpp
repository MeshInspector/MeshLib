#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshCollide.h>
#include <MRMesh/MRMeshComponents.h>
#include <MRVoxels/MROffset.h>
#include <gtest/gtest.h>

namespace MR
{

// returns the number of connected components with not more than maxFaces triangles intersecting a larger component
static int countSmallCollidingComponents( const Mesh& mesh, int maxFaces )
{
    const auto [face2region, numRegions] = MeshComponents::getAllComponentsMap( mesh );
    Vector<int, RegionId> regionFaces( numRegions, 0 );
    for ( auto f : mesh.topology.getValidFaces() )
        ++regionFaces[face2region[f]];
    RegionBitSet colliding( numRegions );
    for ( const auto& p : *findSelfCollidingTriangles( mesh ) )
    {
        const auto a = face2region[p.aFace], b = face2region[p.bFace];
        if ( a == b )
            continue;
        if ( regionFaces[a] <= maxFaces && regionFaces[a] <= regionFaces[b] )
            colliding.set( a );
        if ( regionFaces[b] <= maxFaces && regionFaces[b] <= regionFaces[a] )
            colliding.set( b );
    }
    return (int)colliding.count();
}

TEST( MRMesh, SharpOffsetDeleteSmallCollidingComponents )
{
    // irregular 18-vertex polyhedron, whose zero offset with sharpening gets tiny fragments near the creases
    const VertCoords points = {
        { -0.16324f, 0.027635f, -0.403312f }, { 0.256078f, -0.16713f, 0.472186f }, { 0.598428f, -0.0305243f, 0.305699f },
        { 0.653527f, 0.123001f, 0.520665f }, { 0.565577f, 0.268719f, 0.56379f }, { 0.345204f, 0.403933f, 0.336827f },
        { 0.185074f, 0.438124f, 0.400214f }, { 0.0906841f, 0.37747f, 0.667956f }, { -0.0478251f, 0.280104f, 0.714473f },
        { -0.10331f, 0.151933f, 0.590962f }, { -0.151328f, 0.0587941f, 0.505125f }, { -0.259781f, -0.122413f, 0.466755f },
        { -0.186511f, -0.475161f, 0.617692f }, { -0.018842f, -0.573777f, 0.609779f }, { 0.168236f, -0.479685f, 0.440566f },
        { 0.246968f, -0.541426f, 0.297679f }, { 0.385478f, -0.44406f, 0.251162f }, { 0.562677f, -0.202411f, 0.308097f } };
    Triangulation t;
    for ( int i = 2; i < 18; ++i )
    {
        const VertId a( i ), b( i == 17 ? 2 : i + 1 );
        t.push_back( { 1_v, a, b } );
        t.push_back( { 0_v, b, a } );
    }
    const auto mesh = Mesh::fromTriangles( points, t );

    SharpOffsetParameters params;
    params.voxelSize = 0.021779f;
    params.signDetectionMode = SignDetectionMode::ProjectionNormal;
    params.maxDeletedComponentFaces = 0;
    const auto res0 = sharpOffsetMesh( mesh, 0, params );
    ASSERT_TRUE( res0.has_value() );
    const auto numComps0 = MeshComponents::getNumComponents( *res0 );
    EXPECT_GT( numComps0, size_t( 1 ) );
    EXPECT_GT( countSmallCollidingComponents( *res0, 100 ), 0 );

    params.maxDeletedComponentFaces = 100;
    const auto res1 = sharpOffsetMesh( mesh, 0, params );
    ASSERT_TRUE( res1.has_value() );
    EXPECT_LT( MeshComponents::getNumComponents( *res1 ), numComps0 );
    EXPECT_EQ( countSmallCollidingComponents( *res1, 100 ), 0 );
    EXPECT_TRUE( res1->topology.findHoleRepresentiveEdges().empty() );
}

} //namespace MR
