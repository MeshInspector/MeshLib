#pragma once

#include <version>
#if __cpp_lib_generator >= 202207L
#include <generator>
#else
#include <coroutine>
#if __clang__ && __clang_major__ < 14
// work-around Clang's API requirements
// more info: https://github.com/llvm/llvm-project/issues/47516
namespace std::experimental
{
    template <class Promise = void>
    struct coroutine_handle : std::coroutine_handle<Promise> {};
    template <class R, class... Args>
    struct coroutine_traits : std::coroutine_traits<R, Args...> {};
    using suspend_always = std::suspend_always;
}
#endif
#include <iterator>
#include <utility>
#endif

namespace MR
{

#if __cpp_lib_generator >= 202207L

#pragma message( "Using std::generator" )

template <typename T>
using Generator = std::generator<T>;

#else

#pragma message( "Using MR::Generator" )

/// ...
template <typename T>
class Generator
{
public:
    struct Promise;
    using Coroutine = std::coroutine_handle<Promise>;

    using promise_type = Promise;
    struct Promise
    {
        T value;

        auto get_return_object() { return Generator { Coroutine::from_promise( *this ) }; }
        static std::suspend_always initial_suspend() noexcept { return {}; }
        static std::suspend_always final_suspend() noexcept { return {}; }
        static void return_void() { }
        static void unhandled_exception() { }
        void await_transform() = delete;

        template <std::convertible_to<T> From>
        std::suspend_always yield_value( From&& from ) noexcept
        {
            value = std::forward<From>( from );
            return {};
        }
    };

    class Iterator
    {
    public:
        explicit Iterator( Coroutine coro ) : coro_( coro ) {}
        void operator ++()
        {
            coro_.resume();
        }
        const T& operator *() const
        {
            return coro_.promise().value;
        }
        bool operator ==( std::default_sentinel_t ) const
        {
            return !coro_ || coro_.done();
        }

    private:
        Coroutine coro_;
    };

    explicit Generator( Coroutine coro )
        : coro_( coro )
    {}

    Generator( Generator&& other ) noexcept
        : coro_( other.coro_ )
    {
        other.coro_ = {};
    }
    Generator& operator =( Generator&& other ) noexcept
    {
        if ( this != &other )
        {
            coro_ = other.coro_;
            other.coro_ = {};
        }
        return *this;
    }

    Generator( const Generator& ) = delete;
    Generator& operator =( const Generator& ) = delete;

    ~Generator()
    {
        if ( coro_ )
            coro_.destroy();
    }

    auto begin()
    {
        if ( coro_ )
            coro_.resume();
        return Iterator { coro_ };
    }
    auto end()
    {
        return std::default_sentinel;
    }

private:
    Coroutine coro_;
};

#endif

} // namespace MR
