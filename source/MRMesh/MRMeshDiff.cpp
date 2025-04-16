#include "MRMeshDiff.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRMeshBuilder.h"
#include "MRGTest.h"

namespace MR
{

MeshDiff::MeshDiff( const Mesh & from, const Mesh & to )
{
    MR_TIMER;
    pointsDiff_ = VertCoordsDiff( from.points, to.points );
    topologyDiff_ = MeshTopologyDiff( from.topology, to.topology );
}

void MeshDiff::applyAndSwap( Mesh & m )
{
    MR_TIMER;
    pointsDiff_.applyAndSwap( m.points );
    topologyDiff_.applyAndSwap( m.topology );
}

TEST(MRMesh, MeshDiff)
{
    Triangulation t
    {
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v }
    };
    Mesh mesh0;
    mesh0.topology = MeshBuilder::fromTriangles( t );
    mesh0.points.emplace_back( 0.f, 0.f, 0.f );
    mesh0.points.emplace_back( 1.f, 0.f, 0.f );
    mesh0.points.emplace_back( 1.f, 1.f, 0.f );
    mesh0.points.emplace_back( 0.f, 1.f, 0.f );

    Mesh mesh1 = mesh0;
    mesh1.topology.deleteFace( 1_f );
    mesh1.points.pop_back();

    MeshDiff diff( mesh0, mesh1 );
    EXPECT_EQ( diff.any(), true );
    Mesh m = mesh0;
    EXPECT_EQ( m, mesh0 );
    diff.applyAndSwap( m );
    EXPECT_EQ( diff.any(), true );
    EXPECT_EQ( m, mesh1 );
    diff.applyAndSwap( m );
    EXPECT_EQ( diff.any(), true );
    EXPECT_EQ( m, mesh0 );

    EXPECT_EQ( MeshDiff( m, m ).any(), false );
}

} // namespace MR
