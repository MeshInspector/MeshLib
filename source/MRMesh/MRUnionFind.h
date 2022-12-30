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
    explicit UnionFind( size_t size ) { reset( size ); }
    auto size() const { return roots_.size(); }
    /// reset roots to represent each element as disjoint set of rank 0
    void reset( size_t size )
    {
        roots_.clear();
        roots_.reserve( size );
        for ( I i{ 0 }; i < size; ++i )
            roots_.push_back( i );
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
    /// finds the root of the set containing given element with optimizing data structure updates
    I find( I a )
    {
        return updateRoot_( a, findRootNoUpdate_( a ) );
    }
    /// finds the root of the set containing given element with optimizing data structure in the range [begin, end)
    I findUpdateRange( I a, I begin, I end )
    {
        return updateRootInRange_( a, findRootNoUpdate_( a ), begin, end );
    }
    /// sets the root as the parent of each element, then returns the vector
    const Vector<I, I> & roots()
    {
        for ( I i{ 0 }; i < roots_.size(); ++i )
            updateRoot_( i, findRootNoUpdate_( i ) );
        return roots_;
    }
    /// gets the parents of all elements as in
    const Vector<I, I> & parents() const { return roots_; }

private:
    /// finds the root of the set containing given element without optimizing data structure updates
    I findRootNoUpdate_( I a ) const
    {
        I r = roots_[a];
        for ( I e = a; e != r; r = roots_[e = r] ) {}
        return r;
    }
    /// sets new root \param r for the element \param a and all its ancestors, returns new root \param r
    I updateRoot_( I a, const I r )
    {
        // update parents
        while ( a != r ) 
        { 
            I b = r;
            std::swap( roots_[a], b );
            a = b;
        }
        return r;
    }
    // find the root of given element, and set it as parent for it and other parents
    I updateRoot_( I a ) { return updateRoot_( a, findRootNoUpdate_( a ) ); }
    /// sets new root \param r for the element \param a and all its ancestors if they are in the range [begin, end)
    I updateRootInRange_( I a, const I r, I begin, I end )
    {
        assert( begin < end );
        // update parents
        while ( a != r )
        { 
            if ( a >= begin && a < end )
            {
                I b = r;
                std::swap( roots_[a], b );
                a = b;
            }
            else
                a = roots_[a];
        }
        return r;
    }
    /// roots for each element
    Vector<I, I> roots_;
    /// sizes of each set
    Vector<int, I> sizes_;
};

}
