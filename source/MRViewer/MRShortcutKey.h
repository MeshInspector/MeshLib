#pragma once
#include "MRViewerFwd.h"
#include <compare>
#include <utility>
#include <vector>

namespace MR
{

struct ShortcutKey
{
    int key{ 0 };
    int mod{ 0 };
    /// another key held down when `key` is pressed, e.g. Space in Space+1; 0 if none; matched by its GLFW code, not the keyboard layout;
    /// a shortcut on the held key alone also fires when a chord starts
    int heldKey{ 0 };

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
    /// alternative keys doing the same, e.g. Ctrl+Shift+Z and Ctrl+Y for Redo
    std::vector<ShortcutKey> keys;
    ShortcutCategory category{};

    Shortcut() = default;
    Shortcut( ShortcutKey k, ShortcutCategory c ) : keys{ k }, category( c ) {}
    Shortcut( std::vector<ShortcutKey> ks, ShortcutCategory c ) : keys( std::move( ks ) ), category( c ) {}

    bool operator==( const Shortcut& ) const = default;
};

} //namespace MR
