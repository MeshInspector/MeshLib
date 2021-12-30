#include "MRMeshDelete.h"
#include "MRMeshNormals.h"
#include "MRMesh.h"
#include "MRVector.h"
#include "MRCube.h"
#include "MRTimer.h"
#include "MRMeshBuilder.h"
#include "MRGTest.h"

namespace MR
{

void deleteFace( MeshTopology & topology, FaceId f )
{
    EdgeId e = topology.edgeWithLeft( f );
    assert( e.valid() );
    if ( !e.valid() )
        return;

    // delete the face itself
    topology.setLeft( e, FaceId{} );

    // delete not shared vertices and edges
    const int d = topology.getLeftDegree( e );
    for ( int i = 0; i < d; ++i )
    {
        if ( !topology.right( e ).valid() && topology.prev( e ) == topology.next( e ) )
        {
            // only two edges from e.origin, so this vertex does not belong to any other face
            topology.setOrg( e, VertId{} );
        }
        EdgeId e1 = e;
        e = topology.prev( e.sym() );
        if ( !topology.left( e1.sym() ).valid() )
        {
            // no face to the right of e1, delete it
            topology.splice( topology.prev( e1 ), e1 );
            topology.splice( topology.prev( e1.sym() ), e1.sym() );
        }
    } 
}

void deleteFaces( MeshTopology & topology, const FaceBitSet & fs )
{
    MR_TIMER

    for ( auto f : fs )
        deleteFace( topology, f );
}

void deleteTargetFaces( Mesh& obj, const Vector3f& targetCenter )
{
    MR_TIMER;
    MR_MESH_WRITER( obj );

    auto& topology = obj.topology;
    auto& edgePerFaces = topology.edgePerFace();
    auto& points = obj.points;
    for ( FaceId i{ 0 }; i < edgePerFaces.size(); ++i )
    {
        auto& edge = edgePerFaces[i];
        if ( !edge.valid() )
            continue;
        VertId v0, v1, v2;
        topology.getLeftTriVerts( edge, v0, v1, v2 );
        auto norm = cross( points[v1] - points[v0], points[v2] - points[v0] );
        auto center = ( points[v2] + points[v1] + points[v0] ) / 3.f;
        auto v = dot( norm, targetCenter - center );
        if ( v > 0.f )
        {
            deleteFace( topology, i );
        }
    }
}

void deleteTargetFaces( Mesh & obj, const Mesh & target )
{
    MR_TIMER;
    MR_MESH_WRITER( obj );

    // lets find the center of the tooth root
    Vector3f targetCenter = target.findCenterFromFaces();
    deleteTargetFaces( obj, targetCenter );
}

TEST(MRMesh, DeleteTargetFaces)
{
    Mesh meshObj = makeCube({ 1.f, 1.f, 1.f }, { 0.f, 0.f, 0.f });
    Mesh meshRef = makeCube({ 1.f, 1.f, 1.f }, { -1.f, -1.f, -1.f });

    EXPECT_EQ(meshObj.topology.numValidVerts(), 8);
    EXPECT_EQ(meshObj.topology.numValidFaces(), 12);
    EXPECT_EQ(meshObj.points.size(), 8);

    deleteTargetFaces(meshObj, meshRef);

    EXPECT_EQ(meshObj.topology.numValidVerts(), 7);
    EXPECT_EQ(meshObj.topology.numValidFaces(), 6);
    EXPECT_EQ(meshObj.points.size(), 8);
}

} //namespace MR
