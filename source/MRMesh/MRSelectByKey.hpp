#pragma once

#include <algorithm>
#include <cstddef>

namespace MR
{

namespace SelectByKey
{

/// ranges of at most this many elements are given to sortSmall
constexpr int cSmall = 16;

/// returns the first of the elements in [l, r) not satisfying pred after moving all satisfying ones before it:
/// blocks of elements are scanned for misplaced ones without branches, then the misplaced ones are swapped in pairs
template<typename T, typename P>
T * partition( T * l, T * r, P pred )
{
    constexpr int B = 64;
    unsigned char offL[B], offR[B];
    int nL = 0, nR = 0, sL = 0, sR = 0;
    while ( r - l > 2 * B )
    {
        if ( nL == 0 )
        {
            sL = 0;
            for ( int i = 0; i < B; ++i )
            {
                offL[nL] = (unsigned char)i;
                nL += !pred( l[i] );
            }
        }
        if ( nR == 0 )
        {
            sR = 0;
            for ( int i = 0; i < B; ++i )
            {
                offR[nR] = (unsigned char)( i + 1 );
                nR += pred( *( r - ( i + 1 ) ) );
            }
        }
        const int n = std::min( nL, nR );
        for ( int j = 0; j < n; ++j )
            std::swap( l[offL[sL + j]], *( r - offR[sR + j] ) );
        nL -= n;
        nR -= n;
        sL += n;
        sR += n;
        if ( nL == 0 )
            l += B;
        if ( nR == 0 )
            r -= B;
    }
    // at most two blocks are left
    for ( T * i = l; i != r; ++i )
    {
        const bool lt = pred( *i );
        std::swap( *i, *l );
        l += lt;
    }
    return l;
}

/// orders the elements of [first, last) by key, the ones with equal keys stay in their order
template<typename T, typename K>
void insertionSort( T * first, T * last, K key )
{
    for ( T * i = first + 1; i < last; ++i )
        for ( T * j = i; j > first && key( *j ) < key( *( j - 1 ) ); --j )
            std::swap( *j, *( j - 1 ) );
}

template<typename T, typename K, typename S>
void select( T * first, T * nth, T * last, K key, S sortSmall, bool linear );

/// returns the median of the medians of the groups of 5 elements of [first, last), which are moved to the beginning of the range:
/// at least 3 of every 10 elements are not bigger than it, and at least 3 are not smaller
template<typename T, typename K, typename S>
T * medianOfMedians( T * first, T * last, K key, S sortSmall )
{
    const auto numGroups = ( last - first ) / 5;
    for ( std::ptrdiff_t g = 0; g < numGroups; ++g )
    {
        T * const d = first + 5 * g;
        insertionSort( d, d + 5, key );
        std::swap( first[g], d[2] );
    }
    T * const mid = first + numGroups / 2;
    select( first, mid, first + numGroups, key, sortSmall, true );
    return mid;
}

/// selectByKey; linear makes every pivot the median of medians, which bounds the time whatever the elements are
template<typename T, typename K, typename S>
void select( T * first, T * nth, T * last, K key, S sortSmall, bool linear )
{
    int budget = 2; // partitions with the cheap pivots, about twice the expected number
    for ( auto n = last - first; n > cSmall; n >>= 1 )
        budget += 2;
    while ( last - first > cSmall )
    {
        // unlucky pivots
        linear = linear || --budget < 0;
        const auto med3 = [key]( T * a, T * b, T * c )
        {
            if ( key( *b ) < key( *a ) )
                std::swap( a, b );
            return key( *c ) < key( *b ) ? ( key( *c ) < key( *a ) ? a : c ) : b;
        };
        const auto n = last - first;
        T * const m = first + n / 2;
        T * pivot;
        if ( linear )
            pivot = medianOfMedians( first, last, key, sortSmall );
        else if ( n > 128 )
        {
            const auto s = n / 8;
            pivot = med3( med3( first, first + s, first + 2 * s ), med3( m - s, m, m + s ), med3( last - 1 - 2 * s, last - 1 - s, last - 1 ) );
        }
        else
            pivot = med3( first, m, last - 1 );
        std::swap( *first, *pivot );
        const auto p = key( *first );
        T * const mid = partition( first + 1, last, [p, key]( const T & x ) { return key( x ) < p; } ) - 1;
        std::swap( *first, *mid );
        if ( mid == nth )
            return;
        if ( nth < mid )
        {
            last = mid;
            continue;
        }
        const bool noneLess = mid == first;
        first = mid + 1;
        if ( noneLess || linear )
        {
            // the pivot can be one of many equal elements, which would leave the range one per partition: all of them leave now
            first = partition( first, last, [p, key]( const T & x ) { return !( p < key( x ) ); } );
            if ( nth < first )
                return;
        }
    }
    sortSmall( first, int( last - first ) );
}

} //namespace SelectByKey

/// moves an element of order nth by key to its place, no elements with bigger keys before it and none with smaller keys after it, as std::nth_element does,
/// but every element gets the same place with any standard library: elements with equal keys are not ordered, and only the moves of this file place them;
/// key( element ) returns a number; sortSmall( first, n ) orders n <= SelectByKey::cSmall elements by key
template<typename T, typename K, typename S>
void selectByKey( T * first, T * nth, T * last, K key, S sortSmall )
{
    SelectByKey::select( first, nth, last, key, sortSmall, false );
}

template<typename T, typename K>
void selectByKey( T * first, T * nth, T * last, K key )
{
    SelectByKey::select( first, nth, last, key, [key]( T * d, int n ) { SelectByKey::insertionSort( d, d + n, key ); }, false );
}

} //namespace MR
