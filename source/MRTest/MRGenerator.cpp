#include <MRMesh/MRCube.h>
#include <MRMesh/MRGenerator.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRRingIterator.h>

#include <gtest/gtest.h>

#if __cpp_lib_ranges >= 201911L
#include <ranges>
#endif

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

    auto splitBy = [] ( std::string_view str, std::string_view sep ) -> Generator<std::string_view>
    {
        size_t index = 0, newIndex = str.find( sep );
        while ( newIndex != std::string_view::npos )
        {
            co_yield str.substr( index, newIndex - index );
            index = newIndex + sep.size();
            newIndex = str.find( sep, index );
        }
        co_yield str.substr( index, newIndex - index );
    };

    constexpr auto str { "Lorem ipsum dolor sit amet" };
    auto seq1 = splitBy( str, " " );
    auto it1 = seq1.begin();
    EXPECT_EQ( *it1, "Lorem" ); ++it1;
    EXPECT_EQ( *it1, "ipsum" ); ++it1;
    EXPECT_EQ( *it1, "dolor" ); ++it1;
    EXPECT_EQ( *it1, "sit" ); ++it1;
    EXPECT_EQ( *it1, "amet" ); ++it1;
    EXPECT_EQ( it1, seq1.end() );

    auto seq2 = splitBy( str, ", " );
    auto it2 = seq2.begin();
    EXPECT_EQ( *it2, str ); ++it2;
    EXPECT_EQ( it2, seq2.end() );

#if __cpp_lib_ranges_join_with >= 202202L && __cpp_lib_ranges_to_container >= 202202L
    auto joined =
        splitBy( str, " " )
        | std::views::join_with( std::string_view { ", " } )
        | std::ranges::to<std::string>();
    EXPECT_EQ( joined, "Lorem, ipsum, dolor, sit, amet" );
#endif
}

} // namespace MR
