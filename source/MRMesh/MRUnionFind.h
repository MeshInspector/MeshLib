#pragma once

#include "MRVector.h"

namespace MR
{

/** 
 * \brief Simple union find data structure
 * \tparam I is an id type, e.g. FaceId
 * \ingroup BasicGroup
 */
template <typename I>
class UnionFind
{
public:
    UnionFind() = default;
    UnionFind( size_t size )
    {
        reset( size );
    }
    /// reset roots to represent each element as disjoint set of rank 0
    void reset( size_t size )
    {
        roots_.resize( size );
        for ( I i{ 0 }; i < roots_.size(); ++i )
            roots_[i] = i;
        sizes_.clear();
        sizes_.resize( size, 1 );
    }
    /// unite two elements, returns new common root
    I unite( I first, I second )
    {
        auto firstRoot = getRoot_( first );
        auto secondRoot = getRoot_( second );
        if ( firstRoot == secondRoot )
            return firstRoot;
        /// select root by size for best performance
        if ( sizes_[firstRoot] < sizes_[secondRoot] )
        {
            roots_[firstRoot] = secondRoot;
            sizes_[secondRoot] += sizes_[firstRoot];
            return secondRoot;
        }
        else
        {
            roots_[secondRoot] = firstRoot;
            sizes_[firstRoot] += sizes_[secondRoot];
            return firstRoot;
        }
    }
    /// returns true if given two elements are from one component
    bool united( I first, I second )
    {
        auto firstRoot = getRoot_( first );
        auto secondRoot = getRoot_( second );
        return firstRoot == secondRoot;
    }
    /// finds root of element
    I find( I a )
    {
        return getRoot_( a );
    }
    /// gets roots for all elements
    const Vector<I, I> & roots()
    {
        for ( I i{ 0 }; i < roots_.size(); ++i )
            roots_[i] = getRoot_( i );
        return roots_;
    }
private:
    I getRoot_( I a )
    {
        while ( a != roots_[a] )
        {
            roots_[a] = roots_[roots_[a]];
            a = roots_[a];
        }
        return a;
    }
    /// roots for each element
    Vector<I, I> roots_;
    /// sizes of each set
    Vector<int, I> sizes_;
};

}
