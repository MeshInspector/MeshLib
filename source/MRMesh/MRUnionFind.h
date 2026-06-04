#pragma once

#include "MRVector.h"
#include "MRTimer.h"
#include <atomic>
#include <utility>

namespace MR
{

/**
 * \brief base of union-find data structures: stores only the parent forest and the operations
 * that depend on it alone, so it is shared by the sequential UnionFind and the ParallelUnionFind
 * \tparam I is the identifier of a set's element, e.g. FaceId
 * \ingroup BasicGroup
 */
template <typename I>
class BaseUnionFind
{
public:
    /// the type that can hold the number of elements of the maximal set (e.g. int for FaceId and size_t for VoxelId)
    using SizeType = typename I::ValueType;

    BaseUnionFind() = default;
    BaseUnionFind( const BaseUnionFind & ) = default;
    BaseUnionFind( BaseUnionFind && ) noexcept = default;
    BaseUnionFind & operator =( const BaseUnionFind & ) = default;
    BaseUnionFind & operator =( BaseUnionFind && ) noexcept = default;

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
    }

    /// unites the two sets containing given elements WITHOUT balancing by set size (BaseUnionFind keeps no sizes):
    /// it simply attaches one set's root under the other's, so chained calls can build taller trees.
    /// Prefer UnionFind::unite, which selects the new root by set size, when many unions follow one another.
    /// \return first: new common root, second: true if the union was done (false if already in one set)
    std::pair<I, bool> uniteUnbalanced( I first, I second )
    {
        auto firstRoot = updateRoot_( first );
        auto secondRoot = updateRoot_( second );
        if ( firstRoot == secondRoot )
            return { firstRoot, false };
        parents_[firstRoot] = secondRoot;
        return { secondRoot, true };
    }

    /// returns true if given two elements are from one set
    bool united( I first, I second )
    {
        return updateRoot_( first ) == updateRoot_( second );
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

protected:
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
};

/**
 * \brief Union-find data structure for representing disjoint sets of elements with few very quick operations:
 * 1) union of two sets in one,
 * 2) checking whether two elements pertain to the same set,
 * 3) finding representative element (root) of each set by any set's element.
 * Sequential implementation that selects the root of a united pair by set size for best performance.
 * \tparam I is the identifier of a set's element, e.g. FaceId
 * \ingroup BasicGroup
 */
template <typename I>
class UnionFind : public BaseUnionFind<I>
{
public:
    /// the type that can hold the number of elements of the maximal set (e.g. int for FaceId and size_t for VoxelId)
    using SizeType = typename I::ValueType;

    UnionFind() = default;

    /// creates union-find with given number of elements, each element is the only one in its disjoint set
    explicit UnionFind( size_t size ) { reset( size ); }

    /// adopts the parent forest of a base/parallel union-find and (re)computes set sizes,
    /// so the result supports the full sequential interface (e.g. unite, sizeOfComp)
    explicit UnionFind( BaseUnionFind<I> && base ) : BaseUnionFind<I>( std::move( base ) )
    {
        sizes_.resize( this->parents_.size(), SizeType( 0 ) );
        for ( I i{ size_t( 0 ) }; i < this->parents_.size(); ++i )
            ++sizes_[ this->findRootNoUpdate_( i ) ];
    }

    /// resets union-find to represent given number of elements, each element is the only one in its disjoint set
    void reset( size_t size )
    {
        BaseUnionFind<I>::reset( size );
        sizes_.clear();
        sizes_.resize( size, 1 );
    }

    /// unite two elements,
    /// \return first: new common root, second: true = union was done, false = first and second were already united
    std::pair<I,bool> unite( I first, I second )
    {
        auto firstRoot = this->updateRoot_( first );
        auto secondRoot = this->updateRoot_( second );
        if ( firstRoot == secondRoot )
            return { firstRoot, false };
        /// select root by size for best performance
        if ( sizes_[firstRoot] < sizes_[secondRoot] )
        {
            this->parents_[firstRoot] = secondRoot;
            sizes_[secondRoot] += sizes_[firstRoot];
            return { secondRoot, true };
        }
        else
        {
            this->parents_[secondRoot] = firstRoot;
            sizes_[firstRoot] += sizes_[secondRoot];
            return { firstRoot, true };
        }
    }

    /// returns the number of elements in the set containing given element
    SizeType sizeOfComp( I a ) { return sizes_[ this->find( a ) ]; }

private:
    /// size of each set, contain valid values only for sets' roots
    Vector<SizeType, I> sizes_;
};

/**
 * \brief Union-find that supports lock-free concurrent construction via uniteAtomic().
 * It links by element id (the smaller id becomes the set root), which keeps the forest acyclic without
 * locks; consequently it does not maintain set sizes (use BaseUnionFind::roots() to read the result).
 * \tparam I is the identifier of a set's element, e.g. FaceId
 * \ingroup BasicGroup
 */
template <typename I>
class ParallelUnionFind : public BaseUnionFind<I>
{
public:
    ParallelUnionFind() = default;

    /// creates union-find with given number of elements, each element is the only one in its disjoint set
    explicit ParallelUnionFind( size_t size ) { this->reset( size ); }

    /// thread-safe unite for parallel construction: links by element id (the smaller id becomes the common root)
    /// using lock-free path halving; may be called concurrently for any pair of elements
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
            if ( casStrong_( this->parents_[first], first, second ) )
                return;
            // first is no longer a root (another thread linked it first): retry
        }
    }

private:
    // Lock-free primitives on one parent slot. std::atomic_ref is used where the standard library provides it
    // (MSVC, libstdc++, recent libc++); otherwise we fall back to compiler atomic builtins (e.g. Apple clang,
    // whose libc++ lacks atomic_ref).

    /// relaxed atomic load of a parent slot
    static I loadAtomic_( I& slot )
    {
#ifdef __cpp_lib_atomic_ref
        return I( std::atomic_ref<typename I::ValueType>( slot.get() ).load( std::memory_order_relaxed ) );
#else
        return I( __atomic_load_n( &slot.get(), __ATOMIC_RELAXED ) );
#endif
    }

    /// best-effort path-halving swap (slot: expected -> desired); weak and relaxed, a lost race is harmless
    static void casWeak_( I& slot, I expected, I desired )
    {
        using T = typename I::ValueType;
        T exp = expected.get();
#ifdef __cpp_lib_atomic_ref
        std::atomic_ref<T>( slot.get() ).compare_exchange_weak( exp, desired.get(), std::memory_order_relaxed );
#else
        __atomic_compare_exchange_n( &slot.get(), &exp, desired.get(), true, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
#endif
    }

    /// linking swap (slot: expected -> desired) once; returns true on success, acquire-release on success
    static bool casStrong_( I& slot, I expected, I desired )
    {
        using T = typename I::ValueType;
        T exp = expected.get();
#ifdef __cpp_lib_atomic_ref
        return std::atomic_ref<T>( slot.get() ).compare_exchange_strong( exp, desired.get(), std::memory_order_acq_rel, std::memory_order_relaxed );
#else
        return __atomic_compare_exchange_n( &slot.get(), &exp, desired.get(), false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED );
#endif
    }

    /// lock-free find of the set's root with best-effort path halving
    I findAtomic_( I a )
    {
        for ( ;; )
        {
            I p = loadAtomic_( this->parents_[a] );
            if ( p == a )
                return a; // a is a root
            I gp = loadAtomic_( this->parents_[p] );
            if ( p == gp )
                return p; // p is a root
            // point a to its grandparent (best-effort: a lost race just means another thread already shortened the path)
            casWeak_( this->parents_[a], p, gp );
            a = gp;
        }
    }
};

} //namespace MR
