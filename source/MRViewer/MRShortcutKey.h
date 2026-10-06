#pragma once
#include "MRViewerFwd.h"
#include <compare>
#include <vector>

namespace MR
{

struct ShortcutKey
{
    int key{ 0 };
    int mod{ 0 };

    auto operator<=>( const ShortcutKey& ) const = default;
};

enum class ShortcutCategory : char
{
    Info,
    Edit,
    View,
    Scene,
    Objects,
    Selection,
    Count
};

/// a keyboard shortcut: the keys to press and the category it is listed under
struct Shortcut
{
    ShortcutKey key;
    ShortcutCategory category{};
    /// other keys doing the same, e.g. Ctrl+Y next to Ctrl+Shift+Z for Redo; (key) stays the main one, shown in tooltips
    std::vector<ShortcutKey> extraKeys;
};

} //namespace MR
