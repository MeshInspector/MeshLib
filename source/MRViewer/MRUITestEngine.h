#pragma once

#include "MRViewer/exports.h"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string_view>
#include <string>
#include <variant>

// This is a low-level header for implementing GUIs that can be interacted with programmatically.
// Most likely you don't need to touch this, just use widgets from `MRUIStyle.h`.

namespace MR::UI::TestEngine
{

namespace detail
{
    template <typename T>
    [[nodiscard]] MRVIEWER_API std::optional<T> createValueLow( std::string_view name, T value, T min, T max );

    extern template MRVIEWER_API std::optional<std::int64_t> createValueLow( std::string_view name, std::int64_t value, std::int64_t min, std::int64_t max );
    extern template MRVIEWER_API std::optional<std::uint64_t> createValueLow( std::string_view name, std::uint64_t value, std::uint64_t min, std::uint64_t max );
    extern template MRVIEWER_API std::optional<double> createValueLow( std::string_view name, double value, double min, double max );
}

// Call this every frame when drawing a button you want to track (regardless of whether it returns true of false).
// If this returns true, simulate a button click.
[[nodiscard]] MRVIEWER_API bool createButton( std::string_view name );

// Create a "value" (slider/drag/...).
// `T` must be a scalar; vector support must be implemented manually.
// Pass `min >= max` to disable the range checks.
// If this returns true, use the new value in place of the current one.
template <typename T>
requires std::is_arithmetic_v<T>
[[nodiscard]] std::optional<T> createValue( std::string_view name, T value, T min, T max )
{
    if ( !( min < max ) )
    {
        min = std::numeric_limits<T>::lowest();
        max = std::numeric_limits<T>::max();
    }
    using U = std::conditional_t<std::is_floating_point_v<T>, double, std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>>;

    static_assert(sizeof(T) <= sizeof(U), "The used type is too large.");

    auto ret = detail::createValueLow( name, U( value ), U( min ), U( max ) );
    return ret ? std::optional<T>( T( *ret ) ) : std::nullopt;
}

// Use those to group buttons into named groups.
MRVIEWER_API void pushTree( std::string_view name );
MRVIEWER_API void popTree();

struct Entry;

struct ButtonEntry
{
    // Set this to true to simulate a button click.
    mutable bool simulateClick = false;
};

// For sliders, drags, etc.
struct ValueEntry
{
    template <typename T>
    struct Value
    {
        // The current value.
        T value = 0;

        // Min/max bounds, INCLUSIVE. If none, those are set to the min/max values representable in this type.
        T min = 0;
        T max = 0;

        // Set to override the value.
        mutable std::optional<T> simulatedValue;

        Value() {} // Make `std::variant` below happy.
    };
    using ValueVar = std::variant<Value<std::int64_t>, Value<std::uint64_t>, Value<double>>;
    ValueVar value;
};

struct GroupEntry
{
    // Using `std::map` over `std::unordered_map` to be able to search by `std::string_view` keys directly.
    std::map<std::string, Entry, std::less<>> elems;
};

struct Entry
{
    std::variant<ButtonEntry, ValueEntry, GroupEntry> value;

    // Mostly for internal use.
    // If this is false, the entry will be removed on the next frame.
    bool visitedOnThisFrame = false;
};

// Returns the current entry tree.
[[nodiscard]] MRVIEWER_API const GroupEntry& getRootEntry();

}
