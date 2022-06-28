#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRphmap.h"
#include <string>
#include <functional>
#include <optional>

namespace MR
{

// this class stores two maps:
// 1) shortcut to action
// 2) action name to shortcut
// it can be used to process, customize and print shortcuts
// indifferent to literals register
class MRVIEWER_CLASS ShortcutManager
{
public:
    virtual ~ShortcutManager() = default;

    struct ShortcutCommand
    {
        std::string name; // name of action
        std::function<void()> action;
        bool repeatable = true; // shortcut shall be applied many times while the user holds the keys down
    };

    struct ShortcutKey
    {
        int key{ 0 };
        int mod{ 0 };
    };

    // set shortcut
    // note: one action can have only one shortcut, one shortcut can have only one action
    // if action already has other shortcut, other one will be removed
    MRVIEWER_API virtual void setShortcut( const ShortcutKey& key, const ShortcutCommand& command );

    using ShortcutList = std::vector<std::pair<ShortcutKey, std::string>>;

    // returns cached list of sorted shortcuts
    // if this structure was changed since last call of this function - updates cache
    MRVIEWER_API const ShortcutList& getShortcutList() const;

    enum class Reason 
    {
        KeyDown,  // the user just pressed the keys
        KeyRepeat // the user holds the keys for a long time
    };

    // if given key has action in shortcut map - process it and returns true, otherwise returns false;
    MRVIEWER_API virtual bool processShortcut( const ShortcutKey& key, Reason = Reason::KeyDown ) const;

    // make string form key and returns it
    MRVIEWER_API static std::string getKeyString( const ShortcutKey& key );

    // if action with given name is present in shortcut list - returns it
    MRVIEWER_API std::optional<ShortcutKey> findShortcutByName( const std::string& name ) const;

protected:
    // returns simple map key from key with modifier (alt, ctrl, shift, etc.)
    MRVIEWER_API static int mapKeyFromKeyAndMod( const ShortcutKey& key );
    // returns key with modifier (alt, ctrl, shift, etc.) from simple map key
    static ShortcutKey kayAndModFromMapKey( int mapKey ) { return { mapKey >> 6, mapKey % ( 1 << 6 ) }; }

    using ShourtcutsMap = HashMap<int, ShortcutCommand>;
    using ShourtcutsBackMap = HashMap<std::string, int>;

    ShourtcutsMap map_;
    ShourtcutsBackMap backMap_;

    mutable std::optional<ShortcutList> listCache_;
};

}