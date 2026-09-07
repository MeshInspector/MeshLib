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
// * MCP: `source/MRViewer/MRUiMcp.cpp`.

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

    // Human-readable interaction status. Built by `composeStatus()` in MRUITestEngineControl.cpp —
    // see that function for the set of values. Agents can branch on `status == "available"` or
    // match `status.starts_with("disabled")`.
    std::string status;
};

// Returns the elements of `path` combined into a single string.
[[nodiscard]] MRVIEWER_API std::string pathToString( const std::vector<std::string>& path );

// Returns the contents of `path`, or an error if the path is wrong.
[[nodiscard]] MRVIEWER_API Expected<std::vector<TypedEntry>> listEntries( const std::vector<std::string>& path );

// `listAllEntries` returns this: each element is `(fullPath, entry)` where `fullPath.back() == entry.name`.
using PathedEntry = std::pair<std::vector<std::string>, TypedEntry>;

// Returns every entry in the subtree rooted at `rootPath` as a flat depth-first list. Pass an empty
// `rootPath` to get the whole tree. Groups are included in the list (identifiable by `type == group`)
// and their descendants appear on subsequent rows with `path` extending theirs.
[[nodiscard]] MRVIEWER_API Expected<std::vector<PathedEntry>> listAllEntries( const std::vector<std::string>& rootPath );

// Presses the button at this path.
// Returns empty string on success (click simulated). If the button was drawn disabled, the press is a silent
// no-op and the return is a non-empty status (`"disabled"` / `"disabled: <reason>"`) matching `composeStatus()`.
// `unexpected` is returned only for hard errors (path not found, entry is not a button).
MRVIEWER_API Expected<std::string> pressButton( const std::vector<std::string>& path );

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

// Modifies the value at the `path`.
// Returns empty string on success (write simulated). If the widget was drawn disabled, the write is a silent
// no-op and the return is a non-empty status (`"disabled"` / `"disabled: <reason>"`) matching `composeStatus()`.
// `unexpected` is returned only for hard errors (path not found, wrong type, out-of-range / not-in-allowedValues).
template <typename T>
MRVIEWER_API Expected<std::string> writeValue( const std::vector<std::string>& path, T value );

extern template MRVIEWER_API Expected<std::string> writeValue( const std::vector<std::string>& path, std::int64_t  value );
extern template MRVIEWER_API Expected<std::string> writeValue( const std::vector<std::string>& path, std::uint64_t value );
extern template MRVIEWER_API Expected<std::string> writeValue( const std::vector<std::string>& path, double        value );
extern template MRVIEWER_API Expected<std::string> writeValue( const std::vector<std::string>& path, std::string   value );


} // namespace MR::UI::TestEngine::Control
