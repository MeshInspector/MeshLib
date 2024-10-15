#pragma once

#include "Concat.h"

#include <type_traits>
#include <utility>

/// helper functions to cast types without explicit qualifiers
template <typename T, typename U>
inline auto cast_to( U* from )
{
    if constexpr ( std::is_const_v<U> )
        return reinterpret_cast<const T*>( from );
    else
        return reinterpret_cast<T*>( from );
}

template <typename T, typename U>
inline auto cast_to( U& from )
{
    if constexpr ( std::is_const_v<U> )
        return reinterpret_cast<const T&>( from );
    else
        return reinterpret_cast<T&>( from );
}

template <typename T, typename U>
inline auto cast_to( U&& from )
{
    return reinterpret_cast<T&&>( from );
}

template <typename T>
struct auto_cast_trait { };

/// helper function to cast to an associated type
template <typename U>
inline auto auto_cast( U&& from )
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

/// helper macro to associate source and target types for auto-casting
#define ADD_AUTO_CAST( From, To ) \
template <> struct auto_cast_trait<From> { using target_type = To; }

/// helper macro to associate two types mutually for auto-casting
#define REGISTER_AUTO_CAST2( Type1, Type2 ) \
ADD_AUTO_CAST( Type1, Type2 );              \
ADD_AUTO_CAST( Type2, Type1 );

#define REGISTER_AUTO_CAST( Type ) \
REGISTER_AUTO_CAST2( Type, MR_CONCAT( MR, Type ) )

/// helper macro to cast a function argument from C pointer to C++ reference
/// \code
/// int mrClassGetValue( const MRClass* obj_ )
/// {
///     ARG( obj ); // declares `const Class& obj` variable and casts `obj_` to it
///     return obj.GetValue();
/// }
/// \endcode
#define ARG( X ) \
auto&& X = *auto_cast( MR_CONCAT( X, _ ) )

/// helper macro to cast a function argument from C pointer to C++ pointer
#define ARG_PTR( X ) \
auto&& X = auto_cast( MR_CONCAT( X, _ ) )

/// helper macro to cast a function argument from C pointer to C++ value
#define ARG_VAL( X ) \
auto&& X = auto_cast( MR_CONCAT( X, _ ) )

/// helper macro to cast a function argument from C pointer to C++ reference of specified type
#define ARG_OF( Type, X ) \
auto&& X = *cast_to<Type>( MR_CONCAT( X, _ ) )

/// helper macro to cast a function argument from C pointer to C++ pointer of specified type
#define ARG_PTR_OF( Type, X ) \
auto&& X = cast_to<Type>( MR_CONCAT( X, _ ) )

/// helper macro to cast a function argument from C pointer to C++ value of specified type
#define ARG_VAL_OF( Type, X ) \
auto&& X = cast_to<Type>( MR_CONCAT( X, _ ) )

/// helper macro to auto-cast and return value
#define RETURN( ... ) \
return auto_cast( __VA_ARGS__ )

/// helper function to allocate a new object of auto-detected type
template <typename T, typename U = std::remove_cvref_t<T>>
inline U* new_from( T&& a )
{
    return new U( std::forward<T&&>( a ) );
}

/// helper function to allocate a new object of auto-detected type, auto-cast and return it
#define RETURN_NEW( ... ) \
return auto_cast( new_from( __VA_ARGS__ ) )

/// helper function to allocate a new object of specified type, auto-cast and return it
#define RETURN_NEW_OF( Type, ... ) \
return auto_cast( new_from<Type>( __VA_ARGS__ ) )
