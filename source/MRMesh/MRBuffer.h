#pragma once

#include "MRNoDefInit.h"
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
 * \tparam I type of index (shall be convertible to size_t)
 * \ingroup BasicGroup
 */
template <typename T, typename I>
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
    [[nodiscard]] bool empty() const { return size_.val == 0; }

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

    [[nodiscard]] const_reference operator[]( I i ) const
    {
        assert( i < size_.val );
        return data_[i];
    }
    [[nodiscard]] reference operator[]( I i )
    {
        assert( i < size_.val );
        return data_[i];
    }

    [[nodiscard]] auto data() { return data_.get(); }
    [[nodiscard]] auto data() const { return data_.get(); }
    /// returns the identifier of the back() element
    [[nodiscard]] I backId() const { assert( !empty() ); return I{ size() - 1 }; }
    /// returns backId() + 1
    [[nodiscard]] I endId() const { return I{ size() }; }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return capacity() * sizeof(T); }

private:
    std::unique_ptr<T[]> data_;
    ZeroOnMove<size_t> capacity_, size_;
};

/// given some buffer map and a key, returns the value associated with the key, or default value if key is invalid
template <typename T, typename I>
inline T getAt( const Buffer<NoDefInit<T>, I> & bmap, I key )
{
    return key ? T{bmap[key]} : T{};
}

template <typename T, typename I>
[[nodiscard]] inline auto begin( const Buffer<T, I> & a )
    { return a.data(); }

template <typename T, typename I>
[[nodiscard]] inline auto begin( Buffer<T, I> & a )
    { return a.data(); }

template <typename T, typename I>
[[nodiscard]] inline auto end( const Buffer<T, I> & a )
    { return a.data() + a.size(); }

template <typename T, typename I>
[[nodiscard]] inline auto end( Buffer<T, I> & a )
    { return a.data() + a.size(); }

/// flat map: I -> T
template <typename T, typename I>
struct BMap
{
    Buffer<T, I> b;
    size_t tsize = 0; ///< target size, all values inside b must be less than this value
};

// mapping of mesh elements: old -> new,
// the mapping is tight (or packing) in the sense that there are no unused new elements within [0, (e/f/v).tsize)
struct PackMapping
{
    UndirectedEdgeBMap e;
    FaceBMap f;
    VertBMap v;
};

} // namespace MR
