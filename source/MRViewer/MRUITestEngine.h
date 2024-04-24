#pragma once

#include "MRViewer/exports.h"

#include <map>
#include <string_view>
#include <string>
#include <variant>

// This is a low-level header for implementing GUIs that can be interacted with programmatically.
// Most likely you don't need to touch this, just use widgets from `MRUIStyle.h`.

namespace MR::UI::TestEngine
{

// Call this every frame when drawing a button you want to track (regardless of whether it returns true of false).
// If this returns true, simulate a button click.
[[nodiscard]] MRVIEWER_API bool createButton( std::string_view name );

// Use those to group buttons into named groups.
MRVIEWER_API void pushTree( std::string_view name );
MRVIEWER_API void popTree();

struct Entry;

struct ButtonEntry
{
    // Set this to true to simulate a button click.
    mutable bool simulateClick = false;
};

struct GroupEntry
{
    // Using `std::map` over `std::unordered_map` to be able to search by `std::string_view` keys directly.
    std::map<std::string, Entry, std::less<>> elems;
};

struct Entry
{
    std::variant<ButtonEntry, GroupEntry> value;

    // Mostly for internal use.
    // If this is false, the entry will be removed on the next call.
    bool alive = false;
};

// Returns the current entry tree.
[[nodiscard]] MRVIEWER_API const GroupEntry& getRootEntry();

}
