#pragma once

#include "MRMeshFwd.h"
#include <algorithm>
#include <cassert>
#include <vector>

namespace MR
{

/// the class stores some number of smallest elements from a larger number of candidates
template<typename T>
class FewSmallest
{
public:
    FewSmallest() {}

    /// configure the object to store at most given number of elements
    explicit FewSmallest( size_t maxElms ) { reset( maxElms ); }

    /// clears the content and reconfigure the object to store at most given number of elements
    void reset( size_t maxElms );

    /// returns the maximum number of elements to be stored here
    size_t maxElms() const { return maxElms_; }

    /// returns whether the container is currently empty
    bool empty() const { return heap_.empty(); }

    /// returns current number of stored element
    size_t size() const { return heap_.size(); }

    /// returns whether we have already maximum number of elements stored
    bool full() const { return size() == maxElms(); }

    /// returns the smallest elements found so far
    const std::vector<T> & get() const { return heap_; }

    /// returns the largest among stored smallest elements
    const T & top() const { assert( !heap_.empty() ); return heap_.front(); }

    /// returns the largest among stored smallest elements or given element if this is empty
    const T & topOr( const T & emptyRes ) const { return !heap_.empty() ? heap_.front() : emptyRes; }

    /// considers one more element, storing it if it is within the smallest
    void push( T t );

    /// removes all stored elements
    void clear() { heap_.clear(); }

private:
    std::vector<T> heap_;
    size_t maxElms_ = 0;
};

template<typename T>
void FewSmallest<T>::reset( size_t maxElms )
{
    heap_.clear();
    heap_.reserve( maxElms );
    maxElms_ = maxElms;
}

template<typename T>
void FewSmallest<T>::push( T t )
{
    assert( heap_.size() <= maxElms() );
    if ( heap_.size() == maxElms() )
    {
        if ( t < heap_.front() )
        {
            std::pop_heap( heap_.begin(), heap_.end() );
            assert( t < heap_.back() );
            heap_.back() = std::move( t );
            std::push_heap( heap_.begin(), heap_.end() );
        }
        return;
    }
    heap_.push_back( std::move( t ) );
    std::push_heap( heap_.begin(), heap_.end() );
}

} //namespace MR
