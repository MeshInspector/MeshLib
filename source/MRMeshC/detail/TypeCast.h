#pragma once

#include "Concat.h"

#include <type_traits>
#include <utility>

template <typename T, typename U>
auto cast_to( U* from )
{
    if constexpr ( std::is_const_v<U> )
        return reinterpret_cast<const T*>( from );
    else
        return reinterpret_cast<T*>( from );
}

template <typename T, typename U>
auto cast_to( U& from )
{
    if constexpr ( std::is_const_v<U> )
        return reinterpret_cast<const T&>( from );
    else
        return reinterpret_cast<T&>( from );
}

template <typename T, typename U>
auto cast_to( U&& from )
{
    return reinterpret_cast<T&&>( from );
}

template <typename T>
struct auto_cast_trait { };

template <typename U>
auto auto_cast( U&& from )
{
    using Base = std::remove_const_t<std::remove_pointer_t<std::remove_reference_t<U>>>;
    if constexpr ( std::is_arithmetic_v<Base> )
    {
        return from;
    }
    else
    {
        using T = typename auto_cast_trait<Base>::target_type;
        return cast_to<T>( std::forward<U&&>( from ) );
    }
}

#define ADD_AUTO_CAST( From, To ) \
template <> struct auto_cast_trait<From> { using target_type = To; }

#define REGISTER_AUTO_CAST2( Type1, Type2 ) \
ADD_AUTO_CAST( Type1, Type2 );              \
ADD_AUTO_CAST( Type2, Type1 );

#define REGISTER_AUTO_CAST( Type ) \
REGISTER_AUTO_CAST2( Type, MR_CONCAT( MR, Type ) )

#define ARG( X ) \
auto&& X = *auto_cast( MR_CONCAT( X, _ ) )

#define ARG_PTR( X ) \
auto&& X = auto_cast( MR_CONCAT( X, _ ) )

#define ARG_VAL( X ) \
auto&& X = auto_cast( MR_CONCAT( X, _ ) )

#define ARG_OF( Type, X ) \
auto&& X = *cast_to<Type>( MR_CONCAT( X, _ ) )

#define ARG_PTR_OF( Type, X ) \
auto&& X = cast_to<Type>( MR_CONCAT( X, _ ) )

#define ARG_VAL_OF( Type, X ) \
auto&& X = cast_to<Type>( MR_CONCAT( X, _ ) )

#define RETURN( ... ) \
return auto_cast( __VA_ARGS__ )

template <typename T, typename U = std::remove_cvref_t<T>>
U* new_from( T&& a )
{
    return new U( std::forward<T&&>( a ) );
}

#define RETURN_NEW( ... ) \
return auto_cast( new_from( __VA_ARGS__ ) )

#define RETURN_NEW_OF( Type, ... ) \
return auto_cast( new_from<Type>( __VA_ARGS__ ) )
