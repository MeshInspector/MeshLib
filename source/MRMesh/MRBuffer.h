#pragma once

#include "MRMeshFwd.h"
#include <cassert>
#include <memory>

namespace MR
{

template <typename T>
struct ZeroOnMove
{
    T val = 0;
    constexpr ZeroOnMove() noexcept {}
    constexpr ZeroOnMove( const ZeroOnMove & ) noexcept = delete;
    constexpr ZeroOnMove( ZeroOnMove && z ) noexcept : val( z.val ) { z.val = 0; }
    constexpr ZeroOnMove& operator =( const ZeroOnMove & ) noexcept = delete;
    constexpr ZeroOnMove& operator =( ZeroOnMove && z ) noexcept { val = z.val; z.val = 0; return * this; }
};

/**
 * \brief std::vector<T>-like container that is
 *  1) resized without initialization of its elements,
 *  2) much simplified: no push_back and many other methods
 * \tparam T type of stored elements
 * \ingroup BasicGroup
 */
template <typename T>
class Buffer
{
public:
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    Buffer() = default;
    explicit Buffer( size_t size ) { resize( size ); }

    [[nodiscard]] auto capacity() const { return capacity_.val; }
    [[nodiscard]] auto size() const { return size_.val; }

    void clear() { data_.reset(); capacity_ = {}; size_ = {}; }

    void resize( size_t newSize ) 
    {
        if ( size_.val == newSize )
            return;
        if ( newSize > capacity_.val )
        {
#if __cpp_lib_smart_ptr_for_overwrite >= 202002L
            data_ = std::make_unique_for_overwrite<T[]>( capacity_.val = newSize );
#else
            data_.reset( new T[capacity_.val = newSize] );
#endif
        }
        size_.val = newSize;
    }

    [[nodiscard]] const_reference operator[]( size_t i ) const
    {
        assert( i < size_.val );
        return data_[i];
    }
    [[nodiscard]] reference operator[]( size_t i )
    {
        assert( i < size_.val );
        return data_[i];
    }

    [[nodiscard]] auto data() { return data_.get(); }
    [[nodiscard]] auto data() const { return data_.get(); }

    void swap( Buffer & b ) { data_.swap( b.data_ ); std::swap( capacity_, b.capacity_ ); std::swap( size_, b.size_ ); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return capacity() * sizeof(T); }

private:
    std::unique_ptr<T[]> data_;
    ZeroOnMove<size_t> capacity_, size_;
};

template <typename T, typename I>
[[nodiscard]] inline auto begin( const Buffer<T> & a )
    { return a.data(); }

template <typename T, typename I>
[[nodiscard]] inline auto begin( Buffer<T> & a )
    { return a.data(); }

template <typename T, typename I>
[[nodiscard]] inline auto end( const Buffer<T> & a )
    { return a.data() + a.size(); }

template <typename T, typename I>
[[nodiscard]] inline auto end( Buffer<T> & a )
    { return a.data() + a.size(); }

} // namespace MR
