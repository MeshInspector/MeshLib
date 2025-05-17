#pragma once

#include <limits>
#include <utility>

namespace MR
{

/// minimum of some elements of type T,
/// and the index of where minimum is reached of type I
template<typename T, typename I>
struct MinArg
{
    T val = std::numeric_limits<T>::max();
    I arg;

    auto asPair() const { return std::make_pair( val, arg ); }

    /// changes val and arg if given point is smaller
    void include( const std::pair<T,I>& p )
    {
        if ( p < asPair() )
        {
            val = p.first;
            arg = p.second;
        }
    }

    /// changes val and arg if given point is smaller
    void include( T testVal, I testArg )
    {
        include( std::make_pair( testVal, testArg ) );
    }

    /// changes val and arg if given point is smaller
    void include( const MinArg & s )
    {
        include( s.asPair() );
    }
};

/// maximum of some elements of type T,
/// and the index of where maximum is reached of type I
template<typename T, typename I>
struct MaxArg
{
    T val = std::numeric_limits<T>::lowest();
    I arg;

    auto asPair() const { return std::make_pair( val, arg ); }

    /// changes val and arg if given point is larger
    void include( const std::pair<T,I>& p )
    {
        if ( p > asPair() )
        {
            val = p.first;
            arg = p.second;
        }
    }

    /// changes val and arg if given point is larger
    void include( T testVal, I testArg )
    {
        include( std::make_pair( testVal, testArg ) );
    }

    /// changes val and arg if given point is larger
    void include( const MaxArg & s )
    {
        include( s.asPair() );
    }
};

/// minimum and maximum of some elements of type T,
/// and the indices of where minimum and maximum are reached of type I
template<typename T, typename I>
struct MinMaxArg
{
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::lowest();
    I minArg, maxArg;

    auto minPair() const { return std::make_pair( min, minArg ); }
    auto maxPair() const { return std::make_pair( max, maxArg ); }

    /// changes min(Arg) and max(Arg) if necessary to include given point
    void include( const std::pair<T,I>& p )
    {
        if ( p < minPair() )
        {
            min = p.first;
            minArg = p.second;
        }
        if ( p > maxPair() )
        {
            max = p.first;
            maxArg = p.second;
        }
    }

    /// changes min(Arg) and max(Arg) if necessary to include given point
    void include( T v, I arg )
    {
        return include( std::make_pair( v, arg ) );
    }

    /// changes min(Arg) and max(Arg) if necessary to include given segment
    void include( const MinMaxArg & s )
    {
        // it shall work with default initialized this and s
        if ( s.minPair() < minPair() )
        {
            min = s.min;
            minArg = s.minArg;
        }
        if ( s.maxPair() > maxPair() )
        {
            max = s.max;
            maxArg = s.maxArg;
        }
    }
};

} //namespace MR
