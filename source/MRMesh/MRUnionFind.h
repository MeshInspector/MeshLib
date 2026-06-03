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
            // the exchange succeeds only while first is still a root
            if ( casAtomic_<std::memory_order_acq_rel>( parents_[first], first, second ) )
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

    // Lock-free primitives on one parent slot, used by uniteAtomic()/findAtomic_().
    // std::atomic_ref is used where the standard library provides it (MSVC, libstdc++, recent libc++);
    // otherwise we fall back to compiler atomic builtins (e.g. Apple clang, whose libc++ lacks atomic_ref).
    static I loadAtomic_( I& slot )
    {
#ifdef __cpp_lib_atomic_ref
        return I( std::atomic_ref<typename I::ValueType>( slot.get() ).load( std::memory_order_relaxed ) );
#else
        return I( __atomic_load_n( &slot.get(), __ATOMIC_RELAXED ) );
#endif
    }

    // single attempt slot: expected -> desired with the given success ordering; returns true on success.
    // Used both for the union link (acq_rel) and for best-effort path halving (relaxed); a lost halving
    // race is harmless, so the same primitive serves both with the ordering chosen per call site.
    template <std::memory_order Success>
    static bool casAtomic_( I& slot, I expected, I desired )
    {
        using T = typename I::ValueType;
        T exp = expected.get();
#ifdef __cpp_lib_atomic_ref
        return std::atomic_ref<T>( slot.get() ).compare_exchange_strong( exp, desired.get(), Success, std::memory_order_relaxed );
#else
        constexpr int success = Success == std::memory_order_relaxed ? __ATOMIC_RELAXED : __ATOMIC_ACQ_REL;
        return __atomic_compare_exchange_n( &slot.get(), &exp, desired.get(), false, success, __ATOMIC_RELAXED );
#endif
    }

    /// lock-free find of the set's root with best-effort path halving;
    /// safe to call concurrently with uniteAtomic() running on the same structure
    I findAtomic_( I a )
    {
        for ( ;; )
        {
            I p = loadAtomic_( parents_[a] );
            if ( p == a )
                return a; // a is a root
            I gp = loadAtomic_( parents_[p] );
            if ( p == gp )
                return p; // p is a root
            // point a to its grandparent (best-effort: a lost race just means another thread already shortened the path)
            casAtomic_<std::memory_order_relaxed>( parents_[a], p, gp );
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
