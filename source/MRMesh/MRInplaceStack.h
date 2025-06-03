#pragma once

#if _cpp_lib_inplace_vector >= 202406L

#include <inplace_vector>
#include <stack>

namespace MR
{

template <typename T, size_t N>
using InplaceStack = std::stack<T, std::inplace_vector<T, N>>;

} // namespace MR

#else

#include <array>

namespace MR
{

/// Container class implementing in-place static sized stack.
template <typename T, size_t N>
class InplaceStack
{
public:
    const T& top() const { assert( size_ > 0 ); return data_[size_-1]; }
    T& top() { assert( size_ > 0 ); return data_[size_-1]; }

    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }

    void push( const T& value ) { assert( size_ < N ); data_[size_++] = value; }
    void push( T&& value ) { assert( size_ < N ); data_[size_++] = std::move( value ); }

    void pop() { assert( size_ > 0 ); size_--; }

private:
    std::array<T, N> data_;
    size_t size_ = 0;
};

} // namespace MR

#endif
