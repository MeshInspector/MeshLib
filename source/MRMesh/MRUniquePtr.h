#pragma once

#include <memory>

namespace MR
{

/// This class wraps std::unique_ptr adding copy constructor and copy assignment operator,
/// which do nothing, but allow putting this as a member in copyable classes
template<typename T>
struct UniquePtr : std::unique_ptr<T>
{
    UniquePtr() noexcept = default;
    UniquePtr( const UniquePtr& ) noexcept : std::unique_ptr<T>() {}
    UniquePtr( UniquePtr&& ) noexcept = default;
    UniquePtr& operator =( const UniquePtr& ) noexcept { return *this; }
    UniquePtr& operator =( UniquePtr&& ) noexcept = default;
    UniquePtr& operator =( std::unique_ptr<T>&& b ) noexcept { *static_cast<std::unique_ptr<T>*>(this) = std::move( b ); return *this; }
};

} //namespace MR
