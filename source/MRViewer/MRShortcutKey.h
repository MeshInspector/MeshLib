#pragma once
#include "MRViewerFwd.h"
#include <compare>

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
};

} //namespace MR
