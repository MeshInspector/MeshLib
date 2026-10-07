#include <MRViewer/MRShortcutManager.h>
#include <MRViewer/MRGladGlfw.h>
#include <MRViewer/MRLambdaRibbonItem.h>
#include <MRViewer/MRRibbonSchema.h>
#include <MRMesh/MRSerializer.h>
#include <MRPch/MRJson.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRViewer, ShortcutParseKey )
{
    EXPECT_EQ( ShortcutManager::parseKey( "S" ), GLFW_KEY_S );
    EXPECT_EQ( ShortcutManager::parseKey( "s" ), GLFW_KEY_S );
    EXPECT_EQ( ShortcutManager::parseKey( "," ), GLFW_KEY_COMMA );
    EXPECT_EQ( ShortcutManager::parseKey( "F2" ), GLFW_KEY_F2 );
    EXPECT_EQ( ShortcutManager::parseKey( "F25" ), GLFW_KEY_F25 );
    EXPECT_EQ( ShortcutManager::parseKey( "Num7" ), GLFW_KEY_KP_7 );
    EXPECT_EQ( ShortcutManager::parseKey( "Num 7" ), GLFW_KEY_KP_7 );
    EXPECT_EQ( ShortcutManager::parseKey( "PageUp" ), GLFW_KEY_PAGE_UP );
    EXPECT_EQ( ShortcutManager::parseKey( "ArrowUp" ), GLFW_KEY_UP );
    EXPECT_EQ( ShortcutManager::parseKey( "ForwardDelete" ), GLFW_KEY_DELETE );
#ifndef __EMSCRIPTEN__ // getGlfwKeyDelete() asks the page for is_mac(), which the test harness does not define
    EXPECT_EQ( ShortcutManager::parseKey( "Delete" ), getGlfwKeyDelete() );
#endif

    EXPECT_FALSE( ShortcutManager::parseKey( "" ) );
    EXPECT_FALSE( ShortcutManager::parseKey( " " ) );
    EXPECT_FALSE( ShortcutManager::parseKey( "F0" ) );
    EXPECT_FALSE( ShortcutManager::parseKey( "F26" ) );
    EXPECT_FALSE( ShortcutManager::parseKey( "F1x" ) );
    EXPECT_FALSE( ShortcutManager::parseKey( "Num10" ) );
    EXPECT_FALSE( ShortcutManager::parseKey( "Foo" ) );
}

// every key with a platform-independent textual display name parses back from it
// (GLFW_KEY_DELETE is displayed as "Delete", which parses to the delete key of the platform)
TEST( MRViewer, ShortcutKeyNameRoundTrip )
{
    for ( int key : { GLFW_KEY_A, GLFW_KEY_Z, GLFW_KEY_0, GLFW_KEY_9, GLFW_KEY_COMMA, GLFW_KEY_MINUS, GLFW_KEY_EQUAL,
                      GLFW_KEY_F1, GLFW_KEY_F12, GLFW_KEY_F25, GLFW_KEY_KP_0, GLFW_KEY_KP_9,
                      GLFW_KEY_TAB, GLFW_KEY_HOME, GLFW_KEY_END, GLFW_KEY_PAGE_UP, GLFW_KEY_PAGE_DOWN,
                      GLFW_KEY_PAUSE, GLFW_KEY_CAPS_LOCK, GLFW_KEY_BACKSPACE, GLFW_KEY_ENTER } )
    {
        EXPECT_EQ( ShortcutManager::parseKey( ShortcutManager::getKeyString( key ) ), key ) << "key " << key;
    }
}

TEST( MRViewer, ShortcutParseModifier )
{
    EXPECT_EQ( ShortcutManager::parseModifier( "Shift" ), GLFW_MOD_SHIFT );
    EXPECT_EQ( ShortcutManager::parseModifier( "shift" ), GLFW_MOD_SHIFT );
    EXPECT_EQ( ShortcutManager::parseModifier( "Ctrl" ), GLFW_MOD_CONTROL );
    EXPECT_EQ( ShortcutManager::parseModifier( "Alt" ), GLFW_MOD_ALT );
    EXPECT_EQ( ShortcutManager::parseModifier( "Super" ), GLFW_MOD_SUPER );
#ifndef __EMSCRIPTEN__ // getGlfwModPrimaryCtrl() asks the page for is_mac(), which the test harness does not define
    const auto primary = getGlfwModPrimaryCtrl();
    const auto secondary = primary == GLFW_MOD_SUPER ? GLFW_MOD_CONTROL : GLFW_MOD_SUPER;
    EXPECT_EQ( ShortcutManager::parseModifier( "Primary" ), primary );
    EXPECT_EQ( ShortcutManager::parseModifier( "Secondary" ), secondary );
#endif

    EXPECT_FALSE( ShortcutManager::parseModifier( "" ) );
    EXPECT_FALSE( ShortcutManager::parseModifier( "Cmd" ) );
}

TEST( MRViewer, ShortcutParseShortcutKey )
{
    using SK = ShortcutKey;
    EXPECT_EQ( ShortcutManager::parseShortcutKey( "S" ), ( SK{ GLFW_KEY_S, 0 } ) );
    EXPECT_EQ( ShortcutManager::parseShortcutKey( "Ctrl+Shift+S" ), ( SK{ GLFW_KEY_S, GLFW_MOD_CONTROL | GLFW_MOD_SHIFT } ) );
    EXPECT_EQ( ShortcutManager::parseShortcutKey( " shift + Num 7 " ), ( SK{ GLFW_KEY_KP_7, GLFW_MOD_SHIFT } ) );
    EXPECT_EQ( ShortcutManager::parseShortcutKey( "Ctrl++" ), ( SK{ '+', GLFW_MOD_CONTROL } ) );
    EXPECT_EQ( ShortcutManager::parseShortcutKey( "+" ), ( SK{ '+', 0 } ) );

    EXPECT_FALSE( ShortcutManager::parseShortcutKey( "" ) );
    EXPECT_FALSE( ShortcutManager::parseShortcutKey( "Ctrl+Shift" ) ); // no key
    EXPECT_FALSE( ShortcutManager::parseShortcutKey( "Foo+S" ) );      // unknown modifier
    EXPECT_FALSE( ShortcutManager::parseShortcutKey( "Ctrl+Foo" ) );   // unknown key

    // round trip through getKeyFullString for the modifiers with platform-independent display names
    for ( SK sk : { SK{ GLFW_KEY_S, GLFW_MOD_CONTROL | GLFW_MOD_SHIFT }, SK{ GLFW_KEY_F2, 0 }, SK{ GLFW_KEY_KP_7, GLFW_MOD_SHIFT }, SK{ GLFW_KEY_COMMA, GLFW_MOD_CONTROL } } )
        EXPECT_EQ( ShortcutManager::parseShortcutKey( ShortcutManager::getKeyFullString( sk ) ), sk );
}

TEST( MRViewer, ShortcutParseCategory )
{
    EXPECT_EQ( ShortcutManager::parseCategory( "Info" ), ShortcutCategory::Info );
    EXPECT_EQ( ShortcutManager::parseCategory( "Edit" ), ShortcutCategory::Edit );
    EXPECT_EQ( ShortcutManager::parseCategory( "View" ), ShortcutCategory::View );
    EXPECT_EQ( ShortcutManager::parseCategory( "Scene" ), ShortcutCategory::Scene );
    EXPECT_EQ( ShortcutManager::parseCategory( "Objects" ), ShortcutCategory::Objects );
    EXPECT_EQ( ShortcutManager::parseCategory( "Selection" ), ShortcutCategory::Selection );

    EXPECT_FALSE( ShortcutManager::parseCategory( "" ) );
    EXPECT_FALSE( ShortcutManager::parseCategory( "info" ) );
    EXPECT_FALSE( ShortcutManager::parseCategory( "Count" ) );
}

TEST( MRViewer, ShortcutSeveralKeys )
{
    using SK = ShortcutKey;
    const SK undoKey{ GLFW_KEY_Z, GLFW_MOD_CONTROL };
    const SK redoKey{ GLFW_KEY_Z, GLFW_MOD_CONTROL | GLFW_MOD_SHIFT };
    const SK redoKey2{ GLFW_KEY_Y, GLFW_MOD_CONTROL };

    ShortcutManager sm;
    int undoCount = 0, redoCount = 0;
    sm.setShortcut( { undoKey, ShortcutCategory::Edit }, { "Undo", [&] { ++undoCount; } } );
    // a repeated key is bound once
    sm.setShortcut( { { redoKey, redoKey2, redoKey }, ShortcutCategory::Edit }, { "Redo", [&] { ++redoCount; } } );

    // every key calls its action
    EXPECT_TRUE( sm.processShortcut( redoKey ) );
    EXPECT_TRUE( sm.processShortcut( redoKey2 ) );
    EXPECT_TRUE( sm.processShortcut( undoKey ) );
    EXPECT_FALSE( sm.processShortcut( { GLFW_KEY_Y, 0 } ) );
    EXPECT_EQ( redoCount, 2 );
    EXPECT_EQ( undoCount, 1 );

    // all keys of an action in their order
    EXPECT_EQ( sm.findShortcutsByName( "Redo" ), ( std::vector<SK>{ redoKey, redoKey2 } ) );
    EXPECT_EQ( sm.findShortcutsByName( "Undo" ), std::vector<SK>{ undoKey } );
    EXPECT_TRUE( sm.findShortcutsByName( "Unknown" ).empty() );
    EXPECT_EQ( sm.findShortcutByName( "Redo" ), redoKey );

    // every action is listed once with all its keys, sorted by category and then by the first key
    EXPECT_EQ( sm.getShortcutList(), ( ShortcutManager::ShortcutList{
        { { undoKey, ShortcutCategory::Edit }, "Undo" },
        { { { redoKey, redoKey2 }, ShortcutCategory::Edit }, "Redo" } } ) );
}

TEST( MRViewer, ShortcutSeveralKeysRemoval )
{
    using SK = ShortcutKey;
    const SK redoKey{ GLFW_KEY_Z, GLFW_MOD_CONTROL | GLFW_MOD_SHIFT };
    const SK redoKey2{ GLFW_KEY_Y, GLFW_MOD_CONTROL };
    const SK otherKey{ GLFW_KEY_R, GLFW_MOD_CONTROL };

    ShortcutManager sm;
    int redoCount = 0, otherCount = 0;
    auto setRedo = [&] ( std::vector<SK> keys )
    {
        sm.setShortcut( { std::move( keys ), ShortcutCategory::Edit }, { "Redo", [&] { ++redoCount; } } );
    };

    // setting the shortcut of an action again replaces all its keys
    setRedo( { redoKey, redoKey2 } );
    setRedo( { otherKey } );
    EXPECT_FALSE( sm.processShortcut( redoKey ) );
    EXPECT_FALSE( sm.processShortcut( redoKey2 ) );
    EXPECT_EQ( sm.findShortcutsByName( "Redo" ), std::vector<SK>{ otherKey } );

    // resetting removes the action with all its keys
    setRedo( { redoKey, redoKey2 } );
    sm.resetShortcut( "Redo" );
    sm.resetShortcut( "Unknown" );
    EXPECT_FALSE( sm.processShortcut( redoKey ) );
    EXPECT_FALSE( sm.processShortcut( redoKey2 ) );
    EXPECT_FALSE( sm.findShortcutByName( "Redo" ) );
    EXPECT_TRUE( sm.getShortcutList().empty() );

    // the action losing a key to another action keeps its other keys
    setRedo( { redoKey, redoKey2 } );
    sm.setShortcut( { redoKey, ShortcutCategory::View }, { "Other", [&] { ++otherCount; } } );
    EXPECT_TRUE( sm.processShortcut( redoKey ) );
    EXPECT_TRUE( sm.processShortcut( redoKey2 ) );
    EXPECT_EQ( otherCount, 1 );
    EXPECT_EQ( redoCount, 1 );
    EXPECT_EQ( sm.findShortcutsByName( "Redo" ), std::vector<SK>{ redoKey2 } );
    EXPECT_EQ( sm.getShortcutList(), ( ShortcutManager::ShortcutList{
        { { redoKey2, ShortcutCategory::Edit }, "Redo" },
        { { redoKey, ShortcutCategory::View }, "Other" } } ) );

    // the action losing its last key is removed
    sm.setShortcut( { redoKey2, ShortcutCategory::View }, { "Third", [] {} } );
    EXPECT_FALSE( sm.findShortcutByName( "Redo" ) );
    EXPECT_EQ( sm.getShortcutList().size(), 2 );

    sm.clear();
    EXPECT_FALSE( sm.processShortcut( redoKey ) );
    EXPECT_FALSE( sm.findShortcutByName( "Other" ) );
    EXPECT_TRUE( sm.getShortcutList().empty() );
}

#ifndef __EMSCRIPTEN__ // getGlfwModPrimaryCtrl() asks the page for is_mac(), which the test harness does not define
namespace
{

// reads the "Shortcut" object of an item in items.json
std::optional<MenuItemShortcut> readItemShortcut( const std::string& shortcutJson )
{
    struct Loader : RibbonSchemaLoader
    {
        using RibbonSchemaLoader::readItemsJson_;
    };
    const auto item = std::make_shared<LambdaRibbonItem>( "ShortcutTestItem", [] {} );
    EXPECT_TRUE( RibbonSchemaHolder::addItem( item ) );
    const auto json = deserializeJsonValue(
        R"({ "Items": [ { "Name": "ShortcutTestItem", "Icon": "", "Tooltip": "", "Shortcut": )" + shortcutJson + " } ] }" );
    EXPECT_TRUE( json.has_value() );
    if ( json )
        Loader().readItemsJson_( *json );
    auto res = RibbonSchemaHolder::findItem( item->name() )->shortcut;
    RibbonSchemaHolder::delItem( item );
    return res;
}

} //anonymous namespace

TEST( MRViewer, ShortcutItemKeys )
{
    using SK = ShortcutKey;
    const auto primary = getGlfwModPrimaryCtrl();
    const SK redoKey{ GLFW_KEY_Z, primary | GLFW_MOD_SHIFT };
    const SK ctrlY{ GLFW_KEY_Y, GLFW_MOD_CONTROL };

    // one key
    auto s = readItemShortcut( R"({ "Keys": "Primary+Shift+Z", "Category": "Edit" })" );
    ASSERT_TRUE( s );
    EXPECT_EQ( s->shortcut.keys, std::vector<SK>{ redoKey } );
    EXPECT_EQ( s->shortcut.category, ShortcutCategory::Edit );

    // several keys in their order, but where Cmd is primary (macOS) the keys with Ctrl next to other keys are skipped
    s = readItemShortcut( R"({ "Keys": [ "Primary+Shift+Z", "Ctrl+Y", "Shift+F4" ], "Category": "Edit" })" );
    ASSERT_TRUE( s );
    std::vector<SK> keys{ redoKey };
    if ( primary == GLFW_MOD_CONTROL )
        keys.push_back( ctrlY );
    keys.push_back( { GLFW_KEY_F4, GLFW_MOD_SHIFT } );
    EXPECT_EQ( s->shortcut.keys, keys );

    // the keys with Ctrl are kept on every platform if all keys have it
    s = readItemShortcut( R"({ "Keys": [ "Ctrl+Y", "Ctrl+Tab" ], "Category": "Edit" })" );
    ASSERT_TRUE( s );
    EXPECT_EQ( s->shortcut.keys, ( std::vector<SK>{ ctrlY, { GLFW_KEY_TAB, GLFW_MOD_CONTROL } } ) );
}
#endif

} //namespace MR
