#include <MRMesh/MRVector.h>
#include <MRMesh/MRId.h>
#include <gtest/gtest.h>

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

    EXPECT_EQ( v.at( 1_v ), 5 );
    EXPECT_THROW( (void)v.at( 2_v ), std::out_of_range );

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

} //namespace MR
