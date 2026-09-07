#pragma once
#include "MRViewerFwd.h"

namespace MR
{

struct ShortcutKey
{
    int key{ 0 };
    int mod{ 0 };

    bool operator<( const ShortcutKey& other ) const
    {
        if ( key < other.key )
            return true;
        if ( key == other.key )
            return mod < other.mod;
        return false;
    }
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
