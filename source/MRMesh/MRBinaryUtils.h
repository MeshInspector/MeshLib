#pragma once

#include "MRMeshFwd.h"

#include <array>
#include <variant>

namespace MR
{

/// returns an array with all available variants
template <typename... Args>
constexpr auto makeVariants( std::variant<Args...> )
{
    return std::array<std::variant<Args...>, sizeof...( Args )> { Args{}... };
}
template <typename Variant>
constexpr auto makeVariants()
{
    return makeVariants( Variant{} );
}

/// ...
enum class BinaryDataType
{
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
};

/// ...
using BinaryDataVariant = std::variant<
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    float,
    double
>;

/// ...
using BinaryDataPtrVariant = std::variant<
    uint8_t*,
    uint16_t*,
    uint32_t*,
    uint64_t*,
    int8_t*,
    int16_t*,
    int32_t*,
    int64_t*,
    float*,
    double*
>;

/// returns BinaryDataType enum value corresponding to the given type
template <typename T>
BinaryDataType getBinaryDataType()
{
    static constexpr auto variant = BinaryDataVariant( T() );
    return BinaryDataType( variant.index() );
}

template <typename Visitor>
auto visit( BinaryDataType type, Visitor&& vis )
{
    static constexpr auto variants = makeVariants<BinaryDataVariant>();
    return std::visit( vis, variants.at( (size_t)type ) );
}

template <typename Byte, typename Visitor, typename = std::enable_if_t<std::is_same_v<std::byte, std::remove_const_t<Byte>>>>
auto visit( BinaryDataType type, Byte* data, size_t size, Visitor&& vis )
{
    return visit( type, [=] <typename T> ( T )
    {
        using T_ = std::conditional_t<std::is_const_v<Byte>, const T, T>;
        return vis( reinterpret_cast<T_*>( data ), size / sizeof( T ) );
    } );
}

inline size_t binaryDataSize( BinaryDataType type )
{
    return visit( type, [] <typename T> ( T )
    {
        return sizeof( T );
    } );
}

enum class BinaryRecordType
{
    Scalar,
    RGB,
    RGBA,
};

/// returns number of elements for given record type
inline size_t binaryRecordSize( BinaryRecordType type )
{
    using enum BinaryRecordType;
    switch ( type )
    {
    case Scalar:
        return 1;
    case RGB:
        return 3;
    case RGBA:
        return 4;
    }
    MR_UNREACHABLE
}

/// iterates over the array
template <typename T, typename Func>
void forEach( BinaryRecordType type, T* data, size_t size, Func&& f )
{
    const auto recSize = binaryRecordSize( type );
    for ( auto i = 0u; i < size / recSize; ++i )
        f( i, data + i * recSize, recSize );
}

} // namespace MR
