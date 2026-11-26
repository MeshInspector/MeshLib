#include <MRMesh/MRGlobalRegistration.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshNormals.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRBitSet.h>
#include <gtest/gtest.h>
#include <cmath>

namespace MR
{

namespace
{

// three overlapping parts of a bumpy sphere without symmetries
std::vector<Mesh> makeParts()
{
    auto sphere = makeSphere( { .radius = 10, .numMeshVertices = 5000 } );
    for ( auto & p : sphere.points )
        p *= 1 + 0.2f * std::sin( 0.7f * p.x ) * std::cos( 0.5f * p.y + 1 ) * std::sin( 0.9f * p.z + 2 );

    FaceBitSet regions[3];
    for ( auto & r : regions )
        r.resize( sphere.topology.faceSize() );
    for ( auto f : sphere.topology.getValidFaces() )
    {
        const auto c = sphere.triCenter( f );
        regions[0].set( f, c.x < 3 );
        regions[1].set( f, c.x > -3 && c.y < 3 );
        regions[2].set( f, c.x > -3 && c.y > -3 );
    }
    std::vector<Mesh> res;
    for ( const auto & r : regions )
        res.push_back( sphere.cloneRegion( r ) );
    return res;
}

AffineXf3f testXf( int i )
{
    return AffineXf3f( Matrix3f::rotation( Vector3f( 1.f + i, 2, 3.f - i ).normalized(), 1.0f + i ), Vector3f( 30.f * i, -5, 7 ) );
}

// the parts are cut from one surface, so all of them must get the transformation of the first part
void checkParts( const Vector<MeshOrPointsXf, ObjId> & objs, const GlobalRegistrationResult & res, const std::vector<Mesh> & parts )
{
    EXPECT_EQ( res.xfs[ObjId( 0 )], objs[ObjId( 0 )].xf );
    float maxDist = 0;
    for ( int i = 0; i < 3; ++i )
    {
        EXPECT_EQ( res.groupRoots[ObjId( i )], ObjId( 0 ) );
        for ( const auto & p : parts[i].points )
            maxDist = std::max( maxDist, distance( res.xfs[ObjId( i )]( p ), res.xfs[ObjId( 0 )]( p ) ) );
    }
    EXPECT_LT( maxDist, 0.05f );
}

} // anonymous namespace

TEST( MRMesh, GlobalRegistration )
{
    const auto parts = makeParts();
    const auto ball = makeSphere( { .radius = 2, .numMeshVertices = 500 } );
    const Mesh empty;

    Vector<MeshOrPointsXf, ObjId> objs;
    for ( int i = 0; i < 3; ++i )
        objs.push_back( { MeshOrPoints( parts[i] ), testXf( i ) } );
    objs.push_back( { MeshOrPoints( ball ), testXf( 3 ) } );
    objs.push_back( { MeshOrPoints( empty ), testXf( 4 ) } );

    GlobalRegistrationParams params;
    params.samplingVoxelSize = 0.3f;
    auto res = globalRegistration( objs, params );
    ASSERT_TRUE( res.has_value() );
    checkParts( objs, *res, parts );
    // separate objects keep their transformations
    for ( int i = 3; i < 5; ++i )
    {
        EXPECT_EQ( res->groupRoots[ObjId( i )], ObjId( i ) );
        EXPECT_EQ( res->xfs[ObjId( i )], objs[ObjId( i )].xf );
    }
}

TEST( MRMesh, GlobalRegistrationPoints )
{
    const auto parts = makeParts();
    std::vector<PointCloud> clouds( parts.size() );
    Vector<MeshOrPointsXf, ObjId> objs;
    for ( int i = 0; i < 3; ++i )
    {
        clouds[i].points = parts[i].points;
        clouds[i].normals = computePerVertNormals( parts[i] );
        clouds[i].validPoints = parts[i].topology.getValidVerts();
        objs.push_back( { MeshOrPoints( clouds[i] ), testXf( i ) } );
    }

    GlobalRegistrationParams params;
    params.samplingVoxelSize = 0.3f;
    auto res = globalRegistration( objs, params );
    ASSERT_TRUE( res.has_value() );
    checkParts( objs, *res, parts );
}

} //namespace MR
