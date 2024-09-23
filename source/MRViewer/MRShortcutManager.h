#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRphmap.h"
#include "MRViewer/MRViewerEventsListener.h"
#include <string>
#include <functional>
#include <optional>

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

// this class stores two maps:
// 1) shortcut to action
// 2) action name to shortcut
// it can be used to process, customize and print shortcuts
// indifferent to literals register
class MRVIEWER_CLASS ShortcutManager : public MultiListener<KeyDownListener, KeyRepeatListener>
{
public:
    virtual ~ShortcutManager() = default;

    using ShortcutKey = MR::ShortcutKey;
    using Category = MR::ShortcutCategory;

    struct ShortcutCommand
    {
        Category category;
        std::string name; // name of action
        std::function<void()> action;
        bool repeatable = true; // shortcut shall be applied many times while the user holds the keys down
    };

    inline static const std::string categoryNames[6] = { "Info", "Edit", "View", "Scene", "Objects", "Selection " };

    // set shortcut
    // note: one action can have only one shortcut, one shortcut can have only one action
    // if action already has other shortcut, other one will be removed
    MRVIEWER_API virtual void setShortcut( const ShortcutKey& key, const ShortcutCommand& command );

    using ShortcutList = std::vector<std::tuple<ShortcutKey, Category, std::string>>;

    // returns cached list of sorted shortcuts (sorting by key)
    // if this structure was changed since last call of this function - updates cache
    MRVIEWER_API const ShortcutList& getShortcutList() const;

    enum class Reason 
    {
        KeyDown,  // the user just pressed the keys
        KeyRepeat // the user holds the keys for a long time
    };
    
    // processShortcut does nothing if not enabled
    bool isEnabled() const { return enabled_; }
    void enable( bool on ) { enabled_ = on; }

    // if given key has action in shortcut map - process it and returns true, otherwise returns false;
    MRVIEWER_API virtual bool processShortcut( const ShortcutKey& key, Reason = Reason::KeyDown ) const;

    MRVIEWER_API bool onKeyDown_( int key, int modifier ) override;
    MRVIEWER_API bool onKeyRepeat_( int key, int modifier ) override;

    //make string from strictly one modifier
    MRVIEWER_API static std::string getModifierString( int mod );
    //make string from a key without modifiers, for arrow characters it uses icons font
    MRVIEWER_API static std::string getKeyString( int key );
    // make string from all modifiers and with/without key and returns it
    MRVIEWER_API static std::string getKeyFullString( const ShortcutKey& key, bool respectKey = true );    

    // if action with given name is present in shortcut list - returns it
    MRVIEWER_API std::optional<ShortcutKey> findShortcutByName( const std::string& name ) const;

    // clear all saved shortcuts
    MRVIEWER_API void clear();
protected:
    // returns simple map key from key with modifier (alt, ctrl, shift, etc.)
    // if respectKeyboard is set, key will be mapped using local keyboard settings (only if it is mapped to latin symbol)
    MRVIEWER_API static int mapKeyFromKeyAndMod( const ShortcutKey& key, bool respectKeyboard );
    // returns key with modifier (alt, ctrl, shift, etc.) from simple map key
    static ShortcutKey kayAndModFromMapKey( int mapKey ) { return { mapKey >> 6, mapKey % ( 1 << 6 ) }; }

    using ShourtcutsMap = HashMap<int, ShortcutCommand>;
    using ShourtcutsBackMap = HashMap<std::string, int>;

    bool enabled_{ true };

    ShourtcutsMap map_;
    ShourtcutsBackMap backMap_;

    mutable std::optional<ShortcutList> listCache_;
};

}