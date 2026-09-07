#include <MRViewer/MRShortcutManager.h>
#include <MRViewer/MRGladGlfw.h>
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
#ifndef __EMSCRIPTEN__ // getGlfwKeyDelete() asks the page for is_mac(), which the test harness does not define
    EXPECT_EQ( ShortcutManager::parseKey( "PDelete" ), getGlfwKeyDelete() );
#endif

    EXPECT_FALSE( ShortcutManager::parseKey( "" ) );
    EXPECT_FALSE( ShortcutManager::parseKey( " " ) );
    EXPECT_FALSE( ShortcutManager::parseKey( "F0" ) );
    EXPECT_FALSE( ShortcutManager::parseKey( "F26" ) );
    EXPECT_FALSE( ShortcutManager::parseKey( "F1x" ) );
    EXPECT_FALSE( ShortcutManager::parseKey( "Num10" ) );
    EXPECT_FALSE( ShortcutManager::parseKey( "Foo" ) );
}

// every key with a textual display name parses back from it
TEST( MRViewer, ShortcutKeyNameRoundTrip )
{
    for ( int key : { GLFW_KEY_A, GLFW_KEY_Z, GLFW_KEY_0, GLFW_KEY_9, GLFW_KEY_COMMA, GLFW_KEY_MINUS, GLFW_KEY_EQUAL,
                      GLFW_KEY_F1, GLFW_KEY_F12, GLFW_KEY_F25, GLFW_KEY_KP_0, GLFW_KEY_KP_9,
                      GLFW_KEY_DELETE, GLFW_KEY_TAB, GLFW_KEY_HOME, GLFW_KEY_END, GLFW_KEY_PAGE_UP, GLFW_KEY_PAGE_DOWN,
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
    EXPECT_EQ( ShortcutManager::parseModifier( "PCtrl" ), getGlfwModPrimaryCtrl() );
#endif

    EXPECT_FALSE( ShortcutManager::parseModifier( "" ) );
    EXPECT_FALSE( ShortcutManager::parseModifier( "Cmd" ) );
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

} //namespace MR
