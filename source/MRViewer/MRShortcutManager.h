#pragma once
#include "MRShortcutKey.h"
#include "MRMesh/MRphmap.h"
#include "MRViewer/MRViewerEventsListener.h"
#include <string>
#include <string_view>
#include <functional>
#include <optional>

namespace MR
{

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

    /// what a shortcut does
    struct ShortcutAction
    {
        std::string name; // name of action
        std::function<void()> func;
        bool repeatable = true; // shortcut shall be applied many times while the user holds the keys down
    };

    /// the parameter of the deprecated setShortcut overload, also the internal storage of an action
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
    MRVIEWER_API virtual void setShortcut( const Shortcut& shortcut, const ShortcutAction& action );

    /// deprecated: pass the category in (shortcut) and the rest in ShortcutAction
    [[deprecated( "use setShortcut( Shortcut, ShortcutAction )" )]]
    void setShortcut( const ShortcutKey& key, const ShortcutCommand& command )
        { setShortcut( { key, command.category }, { command.name, command.action, command.repeatable } ); }

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
    MRVIEWER_API static const char* getModifierString( int mod );
    //make string from a key without modifiers, for arrow characters it uses icons font
    MRVIEWER_API static std::string getKeyString( int key );
    // make string from all modifiers and with/without key and returns it
    MRVIEWER_API static std::string getKeyFullString( const ShortcutKey& key, bool respectKey = true );    

    /// parses the name of a key: one printable character ("S", ","), "F1".."F25", "Num0".."Num9",
    /// "Escape", "Enter", "Space", "Tab", "Backspace", "Home", "End", "PageUp", "PageDown", "Up", "Down", "Left", "Right",
    /// "Delete" - the delete key of this platform (Backspace on macOS, see getGlfwKeyDelete), and "ForwardDelete" - the Delete key on every platform;
    /// spaces are ignored, so the output of getKeyString parses back; returns nothing for an unknown name
    MRVIEWER_API static std::optional<int> parseKey( std::string_view name );

    /// parses the name of a modifier (case-insensitive): "Ctrl", "Shift", "Alt", "Super",
    /// "Primary" - the main control of this platform (Cmd on macOS, Ctrl otherwise, see getGlfwModPrimaryCtrl), and "Secondary" - the other of the two;
    /// returns nothing for an unknown name
    MRVIEWER_API static std::optional<int> parseModifier( std::string_view name );

    /// parses the name of a category: one of categoryNames without trailing spaces;
    /// returns nothing for an unknown name
    MRVIEWER_API static std::optional<Category> parseCategory( std::string_view name );

    /// parses a shortcut written as its modifiers and key separated by "+", e.g. "Primary+Shift+S", the inverse of getKeyFullString;
    /// returns nothing if any part is unknown
    MRVIEWER_API static std::optional<ShortcutKey> parseShortcutKey( std::string_view keys );

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