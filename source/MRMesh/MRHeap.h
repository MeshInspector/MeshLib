#pragma once

#include "MRVector.h"
#include "MRMeshFwd.h"
#include <functional>

namespace MR
{

// stores map from element id in [0,size) to T;
// and provides two operations:
// 1) change the value of any element;
// 2) find the element with the largest value
template <typename T, typename I, typename P = std::less<T>>
class Heap
{
public:
    struct Element
    {
        I id;
        T val;
    };

    // constructs heap for given number of elements, assigning given default value to each element
    Heap( int size, T def = {}, P pred = {} );
    // returns the size of the heap
    int size() const { return (int)heap_.size(); }
    // increases the size of the heap by adding elements at the end
    void resize( int size, T def = {} );
    // returns the value associated with given element
    const T & value( I elemId ) const { return heap_[ id2PosInHeap_[ elemId ] ].val; }
    // returns the element with the largest value
    const Element & top() const { return heap_[0]; }
    // sets new value to given element
    void setValue( I elemId, const T & newVal );
    // sets new value to given element, which shall be larger/smaller than the current value
    void setLargerValue( I elemId, const T & newVal );
    void setSmallerValue( I elemId, const T & newVal );
    template<typename U>
    void increaseValue( I elemId, const U & inc ) { setLargerValue( elemId, value( elemId ) + inc ); }
    // sets new value to the current top element, returning its previous value
    Element setTopValue( const T & newVal ) { Element res = top(); setValue( res.id, newVal ); return res; }

private:
    // tests whether heap element at posA is less than posB
    bool less_( int posA, int posB ) const;
    // lifts the element in the queue according to its value
    void lift_( int pos, I elemId );

private:
    std::vector<Element> heap_;
    Vector<int, I> id2PosInHeap_;
    P pred_;
};

template <typename T, typename I, typename P>
Heap<T, I, P>::Heap( int size, T def, P pred )
    : heap_ ( size, { I(), def } )
    , id2PosInHeap_( size )
    , pred_( pred )
{
    for ( I i{0}; i < size; ++i )
    {
        heap_[i].id = i;
        id2PosInHeap_[i] = i;
    }
}

template <typename T, typename I, typename P>
void Heap<T, I, P>::resize( int size, T def )
{
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
    int pos = id2PosInHeap_[ elemId ];
    assert( heap_[pos].id == elemId );
    if ( pred_( newVal, heap_[pos].val ) )
        setSmallerValue( elemId, newVal );
    else if ( pred_( heap_[pos].val, newVal ) )
        setLargerValue( elemId, newVal );
}

template <typename T, typename I, typename P>
void Heap<T, I, P>::setLargerValue( I elemId, const T & newVal )
{
    int pos = id2PosInHeap_[ elemId ];
    assert( heap_[pos].id == elemId );
    assert( !( pred_( newVal, heap_[pos].val ) ) );
    heap_[pos].val = newVal;
    lift_( pos, elemId );
}

template <typename T, typename I, typename P>
void Heap<T, I, P>::lift_( int pos, I elemId )
{
    while ( pos > 0 )
    {
        int parentPos = ( pos - 1 ) / 2;
        if ( !( less_( parentPos, pos ) ) )
            break;
        auto parentId = heap_[parentPos].id;
        assert( id2PosInHeap_[parentId] == parentPos );
        std::swap( heap_[parentPos], heap_[pos] );
        std::swap( parentPos, pos );
        id2PosInHeap_[parentId] = parentPos;
    }
    id2PosInHeap_[elemId] = pos;
}

template <typename T, typename I, typename P>
void Heap<T, I, P>::setSmallerValue( I elemId, const T & newVal )
{
    int pos = id2PosInHeap_[ elemId ];
    assert( heap_[pos].id == elemId );
    assert( !( pred_( heap_[pos].val, newVal ) ) );
    heap_[pos].val = newVal;
    for (;;)
    {
        int child1Pos = 2 * pos + 1;
        if ( child1Pos >= heap_.size() )
            break;
        auto child1Id = heap_[child1Pos].id;
        int child2Pos = 2 * pos + 2;
        if ( child2Pos >= heap_.size() )
        {
            assert( id2PosInHeap_[child1Id] == child1Pos );
            if ( !( less_( child1Pos, pos ) ) )
            {
                std::swap( heap_[child1Pos], heap_[pos] );
                std::swap( child1Pos, pos );
                id2PosInHeap_[child1Id] = child1Pos;
            }
            break;
        }
        auto child2Id = heap_[child2Pos].id;
        if ( !( less_( child1Pos, pos ) ) && !( less_( child1Pos, child2Pos ) ) )
        {
            std::swap( heap_[child1Pos], heap_[pos] );
            std::swap( child1Pos, pos );
            id2PosInHeap_[child1Id] = child1Pos;
        }
        else if ( !( less_( child2Pos, pos ) ) )
        {
            assert( !( less_( child2Pos, child1Pos ) ) );
            std::swap( heap_[child2Pos], heap_[pos] );
            std::swap( child2Pos, pos );
            id2PosInHeap_[child2Id] = child2Pos;
        }
        else
        {
            assert( !( less_( pos, child1Pos ) ) );
            assert( !( less_( pos, child2Pos ) ) );
            break;
        }
    }
    id2PosInHeap_[elemId] = pos;
}

template <typename T, typename I, typename P>
inline bool Heap<T, I, P>::less_( int posA, int posB ) const
{
    const auto & a = heap_[posA];
    const auto & b = heap_[posB];
    if ( pred_( a.val, b.val ) )
        return true;
    if ( pred_( b.val, a.val ) )
        return false;
    return a.id < b.id;
}

} //namespace MR
