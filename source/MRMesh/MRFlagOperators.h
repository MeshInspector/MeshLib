#pragma once

#include <type_traits>

// Generates operators for a enum (at namespace scope).
#define MR_MAKE_FLAG_OPERATORS( T ) MR_MAKE_FLAG_OPERATORS_CUSTOM( static, T )

// Generates operators for a enum (at class scope).
#define MR_MAKE_FLAG_OPERATORS_IN_CLASS( T ) MR_MAKE_FLAG_OPERATORS_CUSTOM( friend, T )

// Generates operators for a enum (with a custom prefix before each function).
#define MR_MAKE_FLAG_OPERATORS_CUSTOM( prefix, T ) \
    [[nodiscard, maybe_unused]] prefix constexpr T operator&( T a, T b ) { return T( ::std::underlying_type_t<T>( a ) & ::std::underlying_type_t<T>( b ) ); } \
    [[nodiscard, maybe_unused]] prefix constexpr T operator|( T a, T b ) { return T( ::std::underlying_type_t<T>( a ) | ::std::underlying_type_t<T>( b ) ); } \
    [[nodiscard, maybe_unused]] prefix constexpr T operator~( T a ) { return T( ~::std::underlying_type_t<T>( a ) ); } \
    [[maybe_unused]] prefix constexpr T &operator&=( T &a, T b ) { return a = a & b; } \
    [[maybe_unused]] prefix constexpr T &operator|=( T &a, T b ) { return a = a | b; } \
    [[nodiscard, maybe_unused]] prefix constexpr T operator*( T a, bool b ) { return b ? a : T{}; } \
    [[nodiscard, maybe_unused]] prefix constexpr T operator*( bool a, T b ) { return a ? b : T{}; } \
    [[maybe_unused]] prefix constexpr T &operator*=( T &a, bool b ) { return a = a * b; }
