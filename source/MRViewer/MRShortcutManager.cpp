#include "MRShortcutManager.h"
#include "MRRibbonConstants.h"
#include "MRImGui.h"
#include "MRGladGlfw.h"
#include <algorithm>
#include <cctype>
#include <string_view>

namespace MR
{

void ShortcutManager::setShortcut( const Shortcut& shortcut, const ShortcutAction& action )
{
    const ShortcutCommand command{ shortcut.category, action.name, action.func, action.repeatable };
    auto newMapKey = mapKeyFromKeyAndMod( shortcut.key, false );
    auto [backMapIt, insertedToBackMap] = backMap_.insert( { command.name,newMapKey } );
    if ( !insertedToBackMap )
    {
        map_.erase( backMapIt->second );
        backMapIt->second = newMapKey;
    }
    
    auto [mapIt, insertedToMap] = map_.insert( { newMapKey,command } );
    if ( !insertedToMap )
    {
        backMap_.erase( mapIt->second.name );
        mapIt->second = command;
    }
    listCache_ = {};
}

const ShortcutManager::ShortcutList& ShortcutManager::getShortcutList() const
{
    if ( listCache_ )
        return *listCache_;

    listCache_ = ShortcutList();
    auto& listRes = *listCache_;
    listRes.reserve( map_.size() );
    for ( const auto& [key, command] : map_ )
        listRes.emplace_back( kayAndModFromMapKey( key ), command.category, command.name );

    std::sort( listRes.begin(), listRes.end(), [] ( const auto& a, const auto& b )
    {
        if ( std::get<Category>( a ) < std::get<Category>( b ) )
            return true;

        if ( std::get<Category>( a ) > std::get<Category>( b ) )
            return false;

        return std::get<ShortcutKey>(a) < std::get<ShortcutKey>(b);
    } );

    return *listCache_;
}

bool ShortcutManager::processShortcut( const ShortcutKey& key, Reason reason ) const
{
    if ( !enabled_ )
        return false;
    auto it = map_.find( mapKeyFromKeyAndMod( key, true ) );
    if ( it != map_.end() && ( reason == Reason::KeyDown || it->second.repeatable ) )
    {
        it->second.action();
        return true;
    }
    return false;
}

bool ShortcutManager::onKeyDown_( int key, int modifier )
{
    return processShortcut( {key, modifier }, Reason::KeyDown );
}


bool ShortcutManager::onKeyRepeat_( int key, int modifier )
{
    return processShortcut( { key, modifier }, Reason::KeyRepeat );
}

const char* ShortcutManager::getModifierString( int mod )
{
    switch ( mod )
    {
    case GLFW_MOD_CONTROL:
        return "Ctrl";
    case GLFW_MOD_ALT:
        return getAltModName();
    case GLFW_MOD_SHIFT:
        return "Shift";
    case GLFW_MOD_SUPER:
        return getSuperModName();
    default:
        return "";
    }
}

std::string ShortcutManager::getKeyString( int key )
{
    if ( key == GLFW_KEY_DELETE )
    {
        return "Delete";
    }
    else if ( key >= GLFW_KEY_F1 && key <= GLFW_KEY_F25 )
    {
        return std::string("F") + std::to_string( key - GLFW_KEY_F1 + 1 );
    }
    else if ( key >= GLFW_KEY_APOSTROPHE && key <= GLFW_KEY_GRAVE_ACCENT )
    {
        return { char( key ) };
    }
    else if ( key >= GLFW_KEY_KP_0 && key <= GLFW_KEY_KP_9 )
    {
        return std::string( "Num " ) + std::to_string( key - GLFW_KEY_KP_0 );
    }
    else if ( key == GLFW_KEY_TAB )
    {
        return std::string( "Tab" );
    }
    else if ( key == GLFW_KEY_HOME )
    {
        return std::string( "Home" );
    }
    else if ( key == GLFW_KEY_END )
    {
        return std::string( "End" );
    }
    else if ( key == GLFW_KEY_PAGE_UP )
    {
        return std::string( "Page Up" );
    }
    else if ( key == GLFW_KEY_PAGE_DOWN )
    {
        return std::string( "Page Down" );
    }
    else if ( key == GLFW_KEY_PAUSE )
    {
        return std::string( "Pause" );
    }
    else if ( key == GLFW_KEY_CAPS_LOCK )
    {
        return std::string( "Caps Lock" );
    }
    else if ( key == GLFW_KEY_BACKSPACE )
    {
        return std::string( "Backspace" );
    }
    else if ( key == GLFW_KEY_ENTER )
    {
        return std::string( "Enter" );
    }
    else if ( key == GLFW_KEY_EQUAL )
    {
        return std::string( "=" );
    }
    else if ( key == GLFW_KEY_MINUS )
    {
        return std::string( "-" );
    }
    else
    {
        switch ( key )
        {
        case GLFW_KEY_UP:
            return "\xef\x81\xa2";
        case GLFW_KEY_DOWN:
            return "\xef\x81\xa3";
        case GLFW_KEY_LEFT:
            return "\xef\x81\xa0";
        case GLFW_KEY_RIGHT:
            return "\xef\x81\xa1";
        default:
            assert( false );
            return "ERROR";
        }
    }
}

std::string ShortcutManager::getKeyFullString( const ShortcutKey& key, bool respectKey )
{
    std::string res;
    if ( key.mod & GLFW_MOD_ALT )
        res += getModifierString( GLFW_MOD_ALT ) + std::string( "+" );
    if ( key.mod & GLFW_MOD_CONTROL )
        res += getModifierString( GLFW_MOD_CONTROL ) + std::string( "+" );
    if ( key.mod & GLFW_MOD_SHIFT )
        res += getModifierString( GLFW_MOD_SHIFT ) + std::string( "+" );
    if ( key.mod & GLFW_MOD_SUPER )
        res += getModifierString( GLFW_MOD_SUPER ) + std::string( "+" );
    if ( respectKey )
        res += getKeyString( key.key );
    return res;
}

std::optional<int> ShortcutManager::parseKey( const std::string& name )
{
    // GLFW codes of letters, digits and punctuation are their upper-case ASCII codes
    if ( name.size() == 1 && std::isprint( (unsigned char)name[0] ) && name[0] != ' ' )
        return std::toupper( (unsigned char)name[0] );

    // "Num 7" of getKeyString and "Num7" are the same key
    std::string s;
    for ( char c : name )
        if ( c != ' ' )
            s += c;

    if ( s == "PDelete" )   return getGlfwKeyDelete();
    if ( s == "Delete" )    return GLFW_KEY_DELETE;
    if ( s == "Backspace" ) return GLFW_KEY_BACKSPACE;
    if ( s == "Enter" || s == "Return" ) return GLFW_KEY_ENTER;
    if ( s == "Escape" )    return GLFW_KEY_ESCAPE;
    if ( s == "Space" )     return GLFW_KEY_SPACE;
    if ( s == "Tab" )       return GLFW_KEY_TAB;
    if ( s == "Home" )      return GLFW_KEY_HOME;
    if ( s == "End" )       return GLFW_KEY_END;
    if ( s == "PageUp" )    return GLFW_KEY_PAGE_UP;
    if ( s == "PageDown" )  return GLFW_KEY_PAGE_DOWN;
    if ( s == "Pause" )     return GLFW_KEY_PAUSE;
    if ( s == "CapsLock" )  return GLFW_KEY_CAPS_LOCK;
    if ( s == "Up" || s == "ArrowUp" )       return GLFW_KEY_UP;
    if ( s == "Down" || s == "ArrowDown" )   return GLFW_KEY_DOWN;
    if ( s == "Left" || s == "ArrowLeft" )   return GLFW_KEY_LEFT;
    if ( s == "Right" || s == "ArrowRight" ) return GLFW_KEY_RIGHT;

    if ( s.size() == 4 && s.starts_with( "Num" ) && std::isdigit( (unsigned char)s[3] ) )
        return GLFW_KEY_KP_0 + ( s[3] - '0' );

    if ( s.size() >= 2 && s[0] == 'F' )
    {
        int n = 0;
        for ( size_t i = 1; i < s.size(); ++i )
        {
            if ( !std::isdigit( (unsigned char)s[i] ) )
                return {};
            n = 10 * n + ( s[i] - '0' );
        }
        if ( n >= 1 && n <= 25 )
            return GLFW_KEY_F1 + ( n - 1 );
    }
    return {};
}

std::optional<int> ShortcutManager::parseModifier( const std::string& name )
{
    std::string s = name;
    std::transform( s.begin(), s.end(), s.begin(), [] ( unsigned char c ) { return (char)std::tolower( c ); } );
    if ( s == "pctrl" ) return getGlfwModPrimaryCtrl();
    if ( s == "ctrl" )  return GLFW_MOD_CONTROL;
    if ( s == "shift" ) return GLFW_MOD_SHIFT;
    if ( s == "alt" )   return GLFW_MOD_ALT;
    if ( s == "super" ) return GLFW_MOD_SUPER;
    return {};
}

std::optional<ShortcutManager::Category> ShortcutManager::parseCategory( const std::string& name )
{
    for ( int i = 0; i < int( Category::Count ); ++i )
    {
        std::string_view categoryName = categoryNames[i];
        while ( categoryName.ends_with( ' ' ) ) // "Selection "
            categoryName.remove_suffix( 1 );
        if ( categoryName == name )
            return Category( i );
    }
    return {};
}

std::optional<ShortcutManager::ShortcutKey> ShortcutManager::findShortcutByName( const std::string& name ) const
{
    auto it = backMap_.find( name );
    if ( it == backMap_.end() )
        return {};
    return kayAndModFromMapKey( it->second );
}

void ShortcutManager::clear()
{
    map_.clear();
    backMap_.clear();
    listCache_ = {};
}

int ShortcutManager::mapKeyFromKeyAndMod( const ShortcutKey& key, [[maybe_unused]] bool respectKeyboard )
{
    int upperKey = key.key;
#ifndef __EMSCRIPTEN__
    if ( respectKeyboard )
    {
        std::string namedKey;
        // map key to char using system keyboard settings
        auto chars = glfwGetKeyName( key.key, glfwGetKeyScancode( key.key ) );
        if ( chars ) // null chars means that mapping failed
            namedKey = std::string( chars );
        // if mapped to latin symbol update `upperKey`
        if ( namedKey.size() == 1 && namedKey[0] >= 'a' && namedKey[0] <= 'z' )
            upperKey = int( namedKey[0] );
    }
#endif

    if ( upperKey >= 'a' && upperKey <= 'z' ) // lower
        upperKey = std::toupper( upperKey );
    return int( upperKey << 6 ) + key.mod;
}

}