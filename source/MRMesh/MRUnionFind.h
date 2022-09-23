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
        auto firstRoot = updateRoot_( first );
        auto secondRoot = updateRoot_( second );
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
        auto firstRoot = updateRoot_( first );
        auto secondRoot = updateRoot_( second );
        return firstRoot == secondRoot;
    }
    /// finds root of element
    I find( I a )
    {
        return updateRoot_( a );
    }
    /// gets roots for all elements
    const Vector<I, I> & roots()
    {
        for ( I i{ 0 }; i < roots_.size(); ++i )
            updateRoot_( i );
        return roots_;
    }
private:
    // find the root of given element, and set it as parent for it and other parents
    I updateRoot_( I a )
    {
        // find root
        I r = roots_[a];
        for ( I e = a; e != r; r = roots_[e = r] ) {}
        // update parents
        while ( a != r ) 
        { 
            I b = r;
            std::swap( roots_[a], b );
            a = b; 
        }
        return r;
    }
    /// roots for each element
    Vector<I, I> roots_;
    /// sizes of each set
    Vector<int, I> sizes_;
};

}
