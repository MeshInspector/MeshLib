#pragma once

#include "MRVector.h"
#include "MRTimer.h"
#include <utility>

namespace MR
{

/** 
 * \brief Union-find data structure for representing disjoin sets of elements with few very quick operations:
 * 1) union of two sets in one,
 * 2) checking whether two elements pertain to the same set,
 * 3) finding representative element (root) of each set by any set's element
 * \tparam I is the identifier of a set's element, e.g. FaceId
 * \ingroup BasicGroup
 */
template <typename I>
class UnionFind
{
public:
    /// the type that can hold the number of elements of the maximal set (e.g. int for FaceId and size_t for VoxelId)
    using SizeType = typename I::ValueType;

    UnionFind() = default;

    /// creates union-find with given number of elements, each element is the only one in its disjoint set
    explicit UnionFind( size_t size ) { reset( size ); }

    /// returns the number of elements in union-find
    auto size() const { return parents_.size(); }

    /// resets union-find to represent given number of elements, each element is the only one in its disjoint set
    void reset( size_t size )
    {
        MR_TIMER;
        parents_.clear();
        parents_.reserve( size );
        for ( I i{ size_t( 0 ) }; i < size; ++i )
            parents_.push_back( i );
        sizes_.clear();
        sizes_.resize( size, 1 );
    }

    /// unite two elements,
    /// \return first: new common root, second: true = union was done, false = first and second were already united
    std::pair<I,bool> unite( I first, I second )
    {
        auto firstRoot = updateRoot_( first );
        auto secondRoot = updateRoot_( second );
        if ( firstRoot == secondRoot )
            return { firstRoot, false };
        /// select root by size for best performance
        if ( sizes_[firstRoot] < sizes_[secondRoot] )
        {
            parents_[firstRoot] = secondRoot;
            sizes_[secondRoot] += sizes_[firstRoot];
            return { secondRoot, true };
        }
        else
        {
            parents_[secondRoot] = firstRoot;
            sizes_[firstRoot] += sizes_[secondRoot];
            return { firstRoot, true };
        }
    }

    /// returns true if given two elements are from one set
    bool united( I first, I second )
    {
        auto firstRoot = updateRoot_( first );
        auto secondRoot = updateRoot_( second );
        return firstRoot == secondRoot;
    }

    /// returns true if given element is the root of some set
    bool isRoot( I a ) const { return parents_[a] == a; }

    /// return parent element of this element, which is equal to given element only for set's root
    I parent( I a ) const { return parents_[a]; }

    /// finds the root of the set containing given element with optimizing data structure updates
    I find( I a ) { return updateRoot_( a ); }

    /// finds the root of the set containing given element with optimizing data structure in the range [begin, end)
    I findUpdateRange( I a, I begin, I end )
    {
        return updateRootInRange_( a, findRootNoUpdate_( a ), begin, end );
    }

    /// sets the root of corresponding set as the parent of each element, then returns the vector
    const Vector<I, I> & roots()
    {
        for ( I i{ size_t( 0 ) }; i < parents_.size(); ++i )
            updateRoot_( i, findRootNoUpdate_( i ) );
        return parents_;
    }

    /// gets the parents of all elements as is
    const Vector<I, I> & parents() const { return parents_; }

    /// returns the number of elements in the set containing given element
    SizeType sizeOfComp( I a ) { return sizes_[ find( a ) ]; }

private:
    /// finds the root of the set containing given element without optimizing data structure updates
    I findRootNoUpdate_( I a ) const
    {
        I r = parents_[a];
        for ( I e = a; e != r; r = parents_[e = r] ) {}
        return r;
    }

    /// sets new root \param r for the element \param a and all its ancestors, returns new root \param r
    I updateRoot_( I a, const I r )
    {
        // update parents
        while ( a != r ) 
        { 
            I b = r;
            std::swap( parents_[a], b );
            a = b;
        }
        return r;
    }

    /// find the root of given element, and set it as parent for it and other parents
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
                std::swap( parents_[a], b );
                a = b;
            }
            else
                a = parents_[a];
        }
        return r;
    }

    /// parent element of each element
    Vector<I, I> parents_;

    /// size of each set, contain valid values only for sets' roots
    Vector<SizeType, I> sizes_;
};

} //namespace MR
