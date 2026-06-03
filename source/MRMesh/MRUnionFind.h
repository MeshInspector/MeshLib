#pragma once

#include "MRVector.h"
#include "MRTimer.h"
#include <atomic>
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

    /// thread-safe version of unite() for parallel construction of the structure:
    /// links by element id (the smaller id becomes the common root) using lock-free path halving;
    /// unlike unite() it does NOT maintain component sizes, so sizeOfComp() is invalid after any uniteAtomic() call
    void uniteAtomic( I first, I second )
    {
        for ( ;; )
        {
            first = findAtomic_( first );
            second = findAtomic_( second );
            if ( first == second )
                return; // already in the same set
            // attach the larger id under the smaller, so the root of a set is always its minimal id
            if ( first < second )
                std::swap( first, second );
            std::atomic_ref<I> parentOfFirst( parents_[first] );
            I expected = first; // the exchange succeeds only while first is still a root
            if ( parentOfFirst.compare_exchange_strong( expected, second,
                    std::memory_order_acq_rel, std::memory_order_relaxed ) )
                return;
            // first is no longer a root (another thread linked it first): retry
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

    /// lock-free find of the set's root with best-effort path halving;
    /// safe to call concurrently with uniteAtomic() running on the same structure
    I findAtomic_( I a )
    {
        for ( ;; )
        {
            std::atomic_ref<I> parentOfA( parents_[a] );
            I p = parentOfA.load( std::memory_order_relaxed );
            if ( p == a )
                return a; // a is a root
            I gp = std::atomic_ref<I>( parents_[p] ).load( std::memory_order_relaxed );
            if ( p == gp )
                return p; // p is a root
            // make a point to its grandparent; a failed exchange only means another thread already shortened the path
            parentOfA.compare_exchange_weak( p, gp, std::memory_order_relaxed );
            a = gp;
        }
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
