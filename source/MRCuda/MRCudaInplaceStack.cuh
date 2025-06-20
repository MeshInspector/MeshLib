#pragma once

#include <utility>

namespace MR::Cuda
{

/// Container class implementing in-place static sized stack.
template <typename T, size_t N>
class InplaceStack
{
public:
    __device__ const T& top() const { assert( size_ > 0 ); return data_[size_-1]; }
    __device__ T& top() { assert( size_ > 0 ); return data_[size_-1]; }

    __device__ bool empty() const { return size_ == 0; }
    __device__ size_t size() const { return size_; }

    __device__ void push( const T& value ) { assert( size_ < N ); data_[size_++] = value; }
    __device__ void push( T&& value ) { assert( size_ < N ); data_[size_++] = std::move( value ); }

    __device__ void pop() { assert( size_ > 0 ); size_--; }

private:
    T data_[N];
    size_t size_ = 0;
};

} // namespace MR::Cuda
