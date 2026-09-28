#pragma once

#include "MRMacros.h"
#include "MRVector.h"
#include "MRTimer.h"
#include <functional>
#include <algorithm>

namespace MR
{

/// \addtogroup BasicGroup
/// \{

/**
 * \brief stores map from element id in[0, size) to T;
 * \details provides two operations:
 * 1) change the value of any element;
 * 2) find the element with the largest value
 */
template <typename T, typename I, typename P = std::less<T>>
class Heap
{
public:
    /// the type that can hold the number of elements of the maximal heap (e.g. int for FaceId and size_t for VoxelId)
    using SizeType = typename I::ValueType;

    struct Element
    {
        I id;
        T val;
    };

    /// constructs an empty heap
    Heap( P pred = {} ) : pred_( pred ) {}

    /// constructs heap for given number of elements, assigning given default value to each element
    explicit Heap( size_t size, T def MR_LIFETIMEBOUND_NESTED = {}, P pred = {} );

    /// constructs heap from given elements (id's shall not repeat and have spaces, but can be arbitrary shuffled)
    explicit Heap( std::vector<Element> elms MR_LIFETIMEBOUND_NESTED, P pred = {} );

    /// returns the size of the heap
    size_t size() const { return heap_.size(); }

    /// increases the size of the heap by adding elements at the end
    void resize( size_t size, T def MR_LIFETIME_CAPTURE_BY_NESTED(this) = {} );

    /// returns the value associated with given element
    const T & value( I elemId ) const MR_LIFETIMEBOUND { return heap_[ id2PosInHeap_[ elemId ] ].val; }

    /// returns the element with the largest value
    const Element & top() const MR_LIFETIMEBOUND { return heap_[0]; }

    /// sets new value to given element
    void setValue( I elemId, const T & newVal MR_LIFETIME_CAPTURE_BY_NESTED(this) );

    /// sets new value to given element, which shall be larger/smaller than the current value
    void setLargerValue( I elemId, const T & newVal MR_LIFETIME_CAPTURE_BY_NESTED(this) );
    void setSmallerValue( I elemId, const T & newVal MR_LIFETIME_CAPTURE_BY_NESTED(this) );
    template<typename U>
    void increaseValue( I elemId, const U & inc ) { setLargerValue( elemId, value( elemId ) + inc ); }

    /// sets new value to the current top element, returning its previous value
    Element setTopValue( const T & newVal MR_LIFETIME_CAPTURE_BY_NESTED(this) ) { Element res = top(); setValue( res.id, newVal ); return res; }

private:
    /// tests whether heap element at posA is less than posB
    bool less_( size_t posA, size_t posB ) const { return less_( heap_[posA], heap_[posB] ); }
    bool less_( const Element & a, const Element & b ) const;

    /// lifts the element in the queue according to its value
    void lift_( size_t pos, I elemId );

private:
    std::vector<Element> heap_;
    Vector<SizeType, I> id2PosInHeap_;
    P pred_;
};

template <typename T, typename I, typename P>
Heap<T, I, P>::Heap( size_t size, T def, P pred )
    : heap_( size, { I(), def } )
    , id2PosInHeap_( size )
    , pred_( pred )
{
    MR_TIMER;
    for ( I i{ size_t( 0 ) }; i < size; ++i )
    {
        heap_[i].id = i;
        id2PosInHeap_[i] = i;
    }
}

template <typename T, typename I, typename P>
Heap<T, I, P>::Heap( std::vector<Element> elms, P pred )
    : heap_( std::move( elms ) )
    , id2PosInHeap_( heap_.size() )
    , pred_( pred )
{
    MR_TIMER;
    std::make_heap( heap_.begin(), heap_.end(), [this]( const Element & a, const Element & b )
        {
            if ( pred_( a.val, b.val ) )
                return true;
            if ( pred_( b.val, a.val ) )
                return false;
            return a.id < b.id;
        }
    );
    for ( size_t i = 0; i < heap_.size(); ++i )
        id2PosInHeap_[heap_[i].id] = SizeType( i );
}

template <typename T, typename I, typename P>
void Heap<T, I, P>::resize( size_t size, T def )
{
    MR_TIMER;
    assert ( heap_.size() == id2PosInHeap_.size() );
    while ( heap_.size() < size )
    {
        I i( heap_.size() );
        heap_.push_back( { i, def } );
        id2PosInHeap_.push_back( i );
        lift_( i, i );
    }
    assert ( heap_.size() == id2PosInHeap_.size() );
}

template <typename T, typename I, typename P>
void Heap<T, I, P>::setValue( I elemId, const T & newVal )
{
    size_t pos = size_t( id2PosInHeap_[ elemId ] );
    assert( heap_[pos].id == elemId );
    if ( pred_( newVal, heap_[pos].val ) )
        setSmallerValue( elemId, newVal );
    else if ( pred_( heap_[pos].val, newVal ) )
        setLargerValue( elemId, newVal );
}

template <typename T, typename I, typename P>
void Heap<T, I, P>::setLargerValue( I elemId, const T & newVal )
{
    size_t pos = size_t( id2PosInHeap_[ elemId ] );
    assert( heap_[pos].id == elemId );
    assert( !( pred_( newVal, heap_[pos].val ) ) );
    heap_[pos].val = newVal;
    lift_( pos, elemId );
}

template <typename T, typename I, typename P>
void Heap<T, I, P>::lift_( size_t pos, I elemId )
{
    assert( heap_[pos].id == elemId );
    Element elem = std::move( heap_[pos] );
    while ( pos > 0 )
    {
        size_t parentPos = ( pos - 1 ) / 2;
        if ( !( less_( heap_[parentPos], elem ) ) )
            break;
        assert( size_t( id2PosInHeap_[heap_[parentPos].id] ) == parentPos );
        heap_[pos] = std::move( heap_[parentPos] );
        id2PosInHeap_[heap_[pos].id] = SizeType( pos );
        pos = parentPos;
    }
    heap_[pos] = std::move( elem );
    id2PosInHeap_[elemId] = SizeType( pos );
}

template <typename T, typename I, typename P>
void Heap<T, I, P>::setSmallerValue( I elemId, const T & newVal )
{
    size_t pos = size_t( id2PosInHeap_[ elemId ] );
    assert( heap_[pos].id == elemId );
    assert( !( pred_( heap_[pos].val, newVal ) ) );
    Element elem{ elemId, newVal };
    // as in std::pop_heap: move the gap down to a leaf filling it with the larger child, then lift the element from there
    for (;;)
    {
        size_t childPos = 2 * pos + 1;
        if ( childPos >= heap_.size() )
            break;
        if ( childPos + 1 < heap_.size() && less_( childPos, childPos + 1 ) )
            ++childPos;
        assert( size_t( id2PosInHeap_[heap_[childPos].id] ) == childPos );
        heap_[pos] = std::move( heap_[childPos] );
        id2PosInHeap_[heap_[pos].id] = SizeType( pos );
        pos = childPos;
    }
    heap_[pos] = std::move( elem );
    lift_( pos, elemId );
}

template <typename T, typename I, typename P>
inline bool Heap<T, I, P>::less_( const Element & a, const Element & b ) const
{
    if ( pred_( a.val, b.val ) )
        return true;
    if ( pred_( b.val, a.val ) )
        return false;
    return a.id < b.id;
}

/// \}

} // namespace MR
