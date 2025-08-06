#include "MRShortcutManager.h"
#include "MRRibbonConstants.h"
#include "MRImGui.h"
#include <GLFW/glfw3.h>

namespace MR
{

void ShortcutManager::setShortcut( const ShortcutKey& key, const ShortcutCommand& command )
{
    auto newMapKey = mapKeyFromKeyAndMod( key, false );
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

std::string ShortcutManager::getModifierString( int mod )
{
    switch ( mod )
    {
    case GLFW_MOD_CONTROL:
        return "Ctrl";
    case GLFW_MOD_ALT:
        return "Alt";
    case GLFW_MOD_SHIFT:
        return "Shift";
    case GLFW_MOD_SUPER:
        return "Command";
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
        res += getModifierString( GLFW_MOD_ALT ) + "+";
    if ( key.mod & GLFW_MOD_CONTROL )
        res += getModifierString( GLFW_MOD_CONTROL ) + "+";
    if ( key.mod & GLFW_MOD_SHIFT )
        res += getModifierString( GLFW_MOD_SHIFT ) + "+";
    if ( key.mod & GLFW_MOD_SUPER )
        res += getModifierString( GLFW_MOD_SUPER ) + "+";
    if ( respectKey )
        res += getKeyString( key.key );
    return res;
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