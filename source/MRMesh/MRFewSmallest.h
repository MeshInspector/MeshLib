#pragma once

#include "MRMeshFwd.h"
#include <cassert>
#include <vector>

namespace MR
{

/// the class stores some number of smallest elements from a larger number of candidates
template<typename T>
class FewSmallest
{
public:
    /// configure the object to store at most given number of elements
    explicit FewSmallest( size_t maxElms );
    /// returns the maximum number of elements to be stored here
    size_t maxElms() const { return heap_.capacity(); }
    /// returns the smallest elements found so far
    const std::vector<T> & get() const { return heap_; }
    /// returns the largest among stored smallest elements
    const T & top() const { assert( !heap_.empty() ); return heap_.front(); }
    /// considers one more element, storing it if it is within the smallest
    void push( T t );
    /// removes all stored elements
    void clear();

private:
    std::vector<T> heap_;
};

template<typename T>
FewSmallest<T>::FewSmallest( size_t maxElms )
{
    assert( maxElms > 0 );
    heap_.reserve( maxElms );
    assert( this->maxElms() == maxElms );
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

template<typename T>
void FewSmallest<T>::clear()
{
#ifndef NDEBUG
    const auto beforeMaxElms = maxElms();
#endif
    heap_.clear();
    assert( beforeMaxElms == maxElms() );
}

} //namespace MR
