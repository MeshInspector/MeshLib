#include <MRMesh/MRCube.h>
#include <MRMesh/MRGenerator.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRRingIterator.h>

#include <gtest/gtest.h>

namespace
{

using namespace MR;

Generator<EdgeId> orgRingCoro( const MeshTopology& top, VertId v )
{
    const auto e0 = top.edgeWithOrg( v );
    if ( !e0.valid() )
        co_return;

    auto e = e0;
    do
    {
        co_yield e;
        e = top.next( e );
    }
    while ( e != e0 );
}

Generator<EdgeId> leftRingCoro( const MeshTopology& top, FaceId f )
{
    const auto e0 = top.edgeWithLeft( f );
    if ( !e0.valid() )
        co_return;

    auto e = e0;
    do
    {
        co_yield e;
        e = top.prev( e.sym() );
    }
    while ( e != e0 );
}

}

namespace MR
{

TEST( MRMesh, Generator )
{
    const auto mesh = makeCube();

    std::vector<EdgeId> edges;
    for ( const auto e : orgRingCoro( mesh.topology, 0_v ) )
        edges.emplace_back( e );
    EXPECT_EQ( edges.size(), 6 );

    size_t i = 0;
    for ( const auto e : orgRing( mesh.topology, 0_v ) )
    {
        EXPECT_EQ( e, edges[i] ) << "at index " << i;
        i++;
    }
    EXPECT_EQ( i, edges.size() );

    edges.clear();
    for ( const auto e : leftRingCoro( mesh.topology, 0_f ) )
        edges.emplace_back( e );
    EXPECT_EQ( edges.size(), 3 );

    i = 0;
    for ( const auto e : leftRing( mesh.topology, 0_f ) )
    {
        EXPECT_EQ( e, edges[i] ) << "at index " << i;
        i++;
    }
    EXPECT_EQ( i, edges.size() );
}

} // namespace MR
