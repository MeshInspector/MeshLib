#pragma once

namespace MR
{

/// \brief simple alternative to boost/iterator_range
/// \defgroup IteratorRange Iterators
/// \ingroup BasicGroup
/// \{
    
template <typename I>
struct IteratorRange
{
    I begin_, end_;
    IteratorRange( I begin, I end ) : begin_( begin ), end_( end ) { }
};

template <typename I>
inline I begin( const IteratorRange<I> & range )
    { return range.begin_; }

template <typename I>
inline I end( const IteratorRange<I> & range )
    { return range.end_; }

/// \}

} //namespace MR
