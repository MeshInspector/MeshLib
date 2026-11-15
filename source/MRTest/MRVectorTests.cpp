#include <MRMesh/MRVector.h>
#include <MRMesh/MRId.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace MR
{

TEST( MRMesh, VectorStdMethods )
{
    Vector<int, VertId> v;
    v.assign( 3, 7 );
    EXPECT_EQ( v, ( Vector<int, VertId>{ 7, 7, 7 } ) );
    v.assign( { 1, 2, 3 } );
    const int arr[] = { 4, 5 };
    v.assign( std::begin( arr ), std::end( arr ) );
    EXPECT_EQ( v, ( Vector<int, VertId>{ 4, 5 } ) );
    EXPECT_GE( v.max_size(), v.size() );

    const int one = 1;
    v.insert( v.begin(), one );          // 1 4 5
    v.insert( v.end(), 6 );              // 1 4 5 6
    v.insert( v.cbegin() + 1, 2, 3 );    // 1 3 3 4 5 6
    v.insert( v.end(), { 8, 9 } );       // 1 3 3 4 5 6 8 9
    v.emplace( v.end() - 2, 7 );         // 1 3 3 4 5 6 7 8 9
    EXPECT_EQ( v, ( Vector<int, VertId>{ 1, 3, 3, 4, 5, 6, 7, 8, 9 } ) );

    v.erase( v.begin() );                // 3 3 4 5 6 7 8 9
    v.erase( v.begin() + 2, v.begin() + 4 ); // 3 3 6 7 8 9
    EXPECT_EQ( v, ( Vector<int, VertId>{ 3, 3, 6, 7, 8, 9 } ) );

    EXPECT_EQ( erase( v, 3 ), 2 );       // 6 7 8 9
    EXPECT_EQ( erase_if( v, []( int x ) { return x % 2 == 0; } ), 2 ); // 7 9
    EXPECT_EQ( v, ( Vector<int, VertId>{ 7, 9 } ) );

    EXPECT_EQ( *v.rbegin(), 9 );
    EXPECT_EQ( *v.crbegin(), 9 );
    EXPECT_EQ( v.rend() - v.rbegin(), 2 );
    EXPECT_EQ( v.crend() - v.crbegin(), 2 );
    EXPECT_EQ( v.cend() - v.cbegin(), 2 );

    int sum = 0;
    for ( int x : std::as_const( v ) )
        sum += x;
    EXPECT_EQ( sum, 16 );
}

TEST( MRMesh, VectorResizeNoInit )
{
    Vector<EdgeId, VertId> v;
    v.resizeNoInit( 1000 );
    EXPECT_EQ( v.size(), 1000 );
    EXPECT_EQ( v.capacity(), 1000 );
    for ( int i = 0; i < 1000; ++i )
        v[VertId( i )] = EdgeId( i );

    v.resizeNoInit( 1001, ReserveMode::Grow );
    EXPECT_EQ( v.size(), 1001 );
    const auto cap = v.capacity();
    EXPECT_GE( cap, 1500 );
    v.resizeNoInit( 1002, ReserveMode::Grow );
    EXPECT_EQ( v.size(), 1002 );
    EXPECT_EQ( v.capacity(), cap );
    for ( int i = 0; i < 1000; ++i )
        EXPECT_EQ( v[VertId( i )], EdgeId( i ) );

    v.resizeNoInit( 5 );
    EXPECT_EQ( v.size(), 5 );
    for ( int i = 0; i < 5; ++i )
        EXPECT_EQ( v[VertId( i )], EdgeId( i ) );
}

namespace
{
struct AlignedRec
{
    static constexpr std::size_t cAllocAlignment = 64;
    int a = 0;
};

struct LaterRec;
struct LaterRecHolder { std::vector<LaterRec, Allocator<LaterRec>> v; }; // the allocator of an incomplete type
struct LaterRec { int a = 0; };
} //anonymous namespace

TEST( MRMesh, VectorAllocAlignment )
{
    using Rec = AlignedRec;
    static_assert( cAllocatorAlignment<Rec> == 64 && cAllocatorAlignment<int> == alignof( int ) );
    const auto aligned = [] ( const Rec * p ) { return reinterpret_cast<std::uintptr_t>( p ) % 64 == 0; };

    Vector<Rec, VertId> v;
    for ( int i = 0; i < 1000; ++i )
    {
        v.push_back( { i } );
        EXPECT_TRUE( aligned( v.data() ) );
    }
    v.resize( 3'000'000 ); // a large block
    EXPECT_TRUE( aligned( v.data() ) );
    EXPECT_EQ( v[VertId( 999 )].a, 999 );
    const auto copy = v;
    EXPECT_TRUE( aligned( copy.data() ) );
    v.clear();
    v.shrink_to_fit();
    v.resize( 3 );
    EXPECT_TRUE( aligned( v.data() ) );

    LaterRecHolder h;
    h.v.push_back( { 5 } );
    EXPECT_EQ( h.v[0].a, 5 );
}

} //namespace MR
