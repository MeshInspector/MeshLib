#pragma once

#include "exports.h"
#include "MRMesh/MRExpected.h"

#include <cassert>
#include <optional>
#include <string>
#include <vector>

namespace MR::UI::TestEngine::Control
{

// Most changes in this file must be synced with:
// * Python: `source/mrviewerpy/MRPythonUiInteraction.cpp`.
// * MCP: `source/MRViewer/MRViewerMcp.cpp`.

enum class EntryType
{
    button,
    group,
    valueInt,
    valueUint,
    valueReal,
    valueString,
};

[[nodiscard]] inline std::string_view toString( EntryType type )
{
    const char* ret = nullptr;
    switch ( type )
    {
        case Control::EntryType::button: ret = "button"; break;
        case Control::EntryType::valueInt: ret = "valueInt"; break;
        case Control::EntryType::valueUint: ret = "valueUint"; break;
        case Control::EntryType::valueReal: ret = "valueReal"; break;
        case Control::EntryType::valueString: ret = "valueString"; break;
        case Control::EntryType::group: ret = "group"; break;
    }
    assert( ret && "Unknown enum." );
    if ( !ret )
        ret = "??";
    return ret;
}

struct TypedEntry
{
    std::string name;
    EntryType type;

    // The widget is currently drawn in a disabled state (greyed out / read-only). Always false for groups.
    bool disabled = false;

    // A blocking popup (e.g. a modal dialog) is currently open and is likely intercepting input from this entry.
    // Heuristic: set only on root-level (top-level) entries; entries inside `pushTree` groups are assumed to
    // live inside the blocking popup itself and are not marked.
    bool blocked = false;
};

// Returns the elements of `path` combined into a single string.
[[nodiscard]] MRVIEWER_API std::string pathToString( const std::vector<std::string>& path );

// Returns the contents of `path`, or an error if the path is wrong.
[[nodiscard]] MRVIEWER_API Expected<std::vector<TypedEntry>> listEntries( const std::vector<std::string>& path );

// Presses the button at this path, or returns an error.
MRVIEWER_API Expected<void> pressButton( const std::vector<std::string>& path );

// Read/write values: (drags, sliders, etc)

template <typename T>
struct Value
{
    T value = 0;
    T min = 0;
    T max = 0;
};
template <>
struct Value<std::string>
{
    std::string value;

    std::optional<std::vector<std::string>> allowedValues;
};
using ValueInt = Value<std::int64_t>;
using ValueUint = Value<std::uint64_t>;
using ValueReal = Value<double>;
using ValueString = Value<std::string>;

// Returns the value at the `path`, or returns an error if the path or type is wrong.
template <typename T>
MRVIEWER_API Expected<Value<T>> readValue( const std::vector<std::string>& path );

extern template MRVIEWER_API Expected<Value<std::int64_t >> readValue( const std::vector<std::string>& path );
extern template MRVIEWER_API Expected<Value<std::uint64_t>> readValue( const std::vector<std::string>& path );
extern template MRVIEWER_API Expected<Value<double       >> readValue( const std::vector<std::string>& path );
extern template MRVIEWER_API Expected<Value<std::string  >> readValue( const std::vector<std::string>& path );

// Modifies the value at the `path`, or returns an error if the path, type or value are wrong.
template <typename T>
MRVIEWER_API Expected<void> writeValue( const std::vector<std::string>& path, T value );

extern template MRVIEWER_API Expected<void> writeValue( const std::vector<std::string>& path, std::int64_t  value );
extern template MRVIEWER_API Expected<void> writeValue( const std::vector<std::string>& path, std::uint64_t value );
extern template MRVIEWER_API Expected<void> writeValue( const std::vector<std::string>& path, double        value );
extern template MRVIEWER_API Expected<void> writeValue( const std::vector<std::string>& path, std::string   value );


} // namespace MR::UI::TestEngine::Control
