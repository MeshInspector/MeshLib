#include <MRMesh/MRGlobalRegistration.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRBitSet.h>
#include <gtest/gtest.h>
#include <cmath>

namespace MR
{

TEST( MRMesh, GlobalRegistration )
{
    // a bumpy sphere without symmetries
    auto sphere = makeSphere( { .radius = 10, .numMeshVertices = 5000 } );
    for ( auto & p : sphere.points )
        p *= 1 + 0.2f * std::sin( 0.7f * p.x ) * std::cos( 0.5f * p.y + 1 ) * std::sin( 0.9f * p.z + 2 );

    // three overlapping parts, and a separate small sphere
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
    const Mesh parts[4] = { sphere.cloneRegion( regions[0] ), sphere.cloneRegion( regions[1] ),
        sphere.cloneRegion( regions[2] ), makeSphere( { .radius = 2, .numMeshVertices = 500 } ) };

    Vector<MeshOrPointsXf, ObjId> objs;
    for ( int i = 0; i < 4; ++i )
        objs.push_back( { MeshOrPoints( parts[i] ),
            AffineXf3f( Matrix3f::rotation( Vector3f( 1.f + i, 2, 3.f - i ).normalized(), 1.0f + i ), Vector3f( 30.f * i, -5, 7 ) ) } );

    GlobalRegistrationParams params;
    params.samplingVoxelSize = 0.3f;
    auto res = globalRegistration( objs, params );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->groupRoots[ObjId( 0 )], ObjId( 0 ) );
    EXPECT_EQ( res->groupRoots[ObjId( 1 )], ObjId( 0 ) );
    EXPECT_EQ( res->groupRoots[ObjId( 2 )], ObjId( 0 ) );
    EXPECT_EQ( res->groupRoots[ObjId( 3 )], ObjId( 3 ) );
    EXPECT_EQ( res->xfs[ObjId( 0 )], objs[ObjId( 0 )].xf );
    EXPECT_EQ( res->xfs[ObjId( 3 )], objs[ObjId( 3 )].xf );

    // the parts are cut from one mesh, so all of them must get the transformation of the first part
    for ( int i = 1; i < 3; ++i )
        for ( const auto & p : parts[i].points )
            EXPECT_LT( distance( res->xfs[ObjId( i )]( p ), res->xfs[ObjId( 0 )]( p ) ), 0.05f );
}

} //namespace MR
