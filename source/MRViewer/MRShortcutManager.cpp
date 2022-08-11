#include "MRShortcutManager.h"
#include "MRRibbonConstants.h"
#include "imgui.h"
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
    auto it = map_.find( mapKeyFromKeyAndMod( key ) );
    if ( it != map_.end() && ( reason == Reason::KeyDown || it->second.repeatable ) )
    {
        it->second.action();
        return true;
    }
    return false;
}

std::string ShortcutManager::getKeyString( const ShortcutKey& key, bool respectLastKey )
{
    std::string res;
    if ( key.mod & GLFW_MOD_ALT )
        res += "Alt+";
    if ( key.mod & GLFW_MOD_CONTROL )
        res += "Ctrl+";
    if ( key.mod & GLFW_MOD_SHIFT )
        res += "Shift+";

    if ( key.key == GLFW_KEY_DELETE )
    {
        res += "Delete";
    }
    else if ( key.key >= GLFW_KEY_F1 && key.key <= GLFW_KEY_F25 )
    {
        res += "F";
        res += std::to_string( key.key - GLFW_KEY_F1 + 1 );
    }
    else if ( respectLastKey && key.key >= GLFW_KEY_APOSTROPHE && key.key <= GLFW_KEY_GRAVE_ACCENT )
    {
        res += char( key.key );
    }
    else if ( respectLastKey )
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
        
            break;
        default:
            assert( false );
            res += "ERROR";
            break;
        }
    }
    return res;
}

// calculates paddings needed for drawing shortcut key with modifiers

float ShortcutManager::getKeyPaddings( const ShortcutKey& key, float scaling )
{
    const auto& style = ImGui::GetStyle();
    float res = 2 * style.FramePadding.x + 3 * style.ItemInnerSpacing.x;;
    if ( key.mod & GLFW_MOD_ALT )
        res += 2 * style.FramePadding.x + 2 * style.ItemInnerSpacing.x;
    if ( key.mod & GLFW_MOD_CONTROL )
        res += 2 * style.FramePadding.x + 2 * style.ItemInnerSpacing.x;
    if ( key.mod & GLFW_MOD_SHIFT )
        res += 2 * style.FramePadding.x + 2 * style.ItemInnerSpacing.x;

    if ( key.key != GLFW_KEY_DELETE && key.key <= GLFW_KEY_F1 && key.key >= GLFW_KEY_F25 )
        res += 2 * cButtonPadding * scaling;

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