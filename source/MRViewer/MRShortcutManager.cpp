#include "MRShortcutManager.h"
#include <GLFW/glfw3.h>

namespace MR
{

void ShortcutManager::setShortcut( const ShortcutKey& key, const ShortcutCommand& command )
{
    auto newMapKey = mapKeyFromKeyAndMod( key );
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

const ShortcutManager::ShortcutList& ShortcutManager::getShortcutList( std::optional< std::function<int( const ShortcutKey& )> > sortByCathegoryCallback ) const
{
    if ( listCache_ )
        return *listCache_;

    listCache_ = ShortcutList();
    auto& listRes = *listCache_;
    listRes.reserve( map_.size() );
    for ( const auto& [key, command] : map_ )
        listRes.emplace_back( kayAndModFromMapKey( key ), command.name );

    std::sort( listRes.begin(), listRes.end(), [sortByCathegoryCallback] ( const auto& a, const auto& b )
    {
        if ( sortByCathegoryCallback )
        {
            int aCat = ( *sortByCathegoryCallback )( a.first );
            int bCat = ( *sortByCathegoryCallback )( b.first );
            if ( aCat < bCat)
                return true;
            if ( aCat > bCat )
                return false;
        }

        return a.first < b.first;
    } );

    return *listCache_;
}

bool ShortcutManager::processShortcut( const ShortcutKey& key, Reason reason ) const
{
    auto it = map_.find( mapKeyFromKeyAndMod( key ) );
    if ( it != map_.end() && ( reason == Reason::KeyDown || it->second.repeatable ) )
    {
        it->second.action();
        return true;
    }
    return false;
}

std::string ShortcutManager::getKeyString( const ShortcutKey& key )
{
    std::string res;
    if ( key.mod & GLFW_MOD_ALT )
        res += "Alt+";
    if ( key.mod & GLFW_MOD_CONTROL )
        res += "Ctrl+";
    if ( key.mod & GLFW_MOD_SHIFT )
        res += "Shift+";

    if ( key.key >= GLFW_KEY_APOSTROPHE && key.key <= GLFW_KEY_GRAVE_ACCENT )
        res += char( key.key );
    else if ( key.key >= GLFW_KEY_F1 && key.key <= GLFW_KEY_F25 )
    {
        res += "F";
        res += std::to_string( key.key - GLFW_KEY_F1 + 1 );
    }
    else
    {
        switch ( key.key )
        {
        case GLFW_KEY_UP:
            res += "Up";
            break;
        case GLFW_KEY_DOWN:
            res += "Down";
            break;
        case GLFW_KEY_LEFT:
            res += "Left";
            break;
        case GLFW_KEY_RIGHT:
            res += "Right";
            break;
        case GLFW_KEY_DELETE:
            res += "Delete";
            break;
        default:
            assert( false );
            res += "ERROR";
            break;
        }
    }
    return res;
}

std::optional<ShortcutManager::ShortcutKey> ShortcutManager::findShortcutByName( const std::string& name ) const
{
    auto it = backMap_.find( name );
    if ( it == backMap_.end() )
        return {};
    return kayAndModFromMapKey( it->second );
}

int ShortcutManager::mapKeyFromKeyAndMod( const ShortcutKey& key )
{
    int upperKey = key.key;
    if ( upperKey >= 'a' && upperKey <= 'z' ) // lower
        upperKey = std::toupper( upperKey );
    return int( upperKey << 6 ) + key.mod;
}

}