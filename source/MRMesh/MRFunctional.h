#pragma once

// based on https://github.com/TartanLlama/function_ref

#include <functional>
#include <utility>

namespace MR
{

template <typename F>
class FunctionRef;

/// \brief Simplified implementation of std::function_ref from C++26
/// \details
/// FunctionRef can be used as a lightweight replacement for std::function in cases when:
///  - a callable exists (there's no equivalent to `std::function( nullptr )`)
///  - a callable is passed to a function that calls it immediately or passes to another function of the same kind
template <typename R, typename... Args>
class FunctionRef<R ( Args... )>
{
public:
    constexpr FunctionRef() noexcept = delete;

    constexpr FunctionRef( const FunctionRef& rhs ) noexcept = default;

    template <
        typename F,
        std::enable_if_t<
            std::is_invocable_r_v<R, F&&, Args...>
        > * = nullptr
    >
    constexpr FunctionRef( F&& f ) noexcept
    {
        obj_ = reinterpret_cast<void *>( std::addressof( f ) );
        callback_ = [] ( void *obj, Args... args ) -> R
        {
            return std::invoke(
                *reinterpret_cast<std::add_pointer_t<F>>( obj ),
                std::forward<Args>( args )...
            );
        };
    }

    constexpr FunctionRef& operator =( const FunctionRef& rhs ) noexcept = default;

    template <
        typename F,
        std::enable_if_t<
            std::is_invocable_r_v<R, F&&, Args...>
        >* = nullptr
    >
    constexpr FunctionRef& operator =( F&& f ) noexcept
    {
        obj_ = reinterpret_cast<void *>( std::addressof( f ) );
        callback_ = [] ( void *obj, Args... args ) -> R
        {
            return std::invoke(
                *reinterpret_cast<std::add_pointer_t<F>>( obj ),
                std::forward<Args>( args )...
            );
        };
        return *this;
    }

    constexpr void swap( FunctionRef& rhs ) noexcept
    {
        std::swap( obj_, rhs.obj_ );
        std::swap( callback_, rhs.callback_ );
    }

    R operator ()( Args... args ) const
    {
        return callback_( obj_, std::forward<Args>( args )... );
    }

private:
    void *obj_ = nullptr;
    R (*callback_)( void*, Args... ) = nullptr;
};

template <typename R, typename... Args>
constexpr void swap( FunctionRef<R ( Args... )>& lhs, FunctionRef<R ( Args... )>& rhs ) noexcept
{
    lhs.swap( rhs );
}

} // namespace MR
