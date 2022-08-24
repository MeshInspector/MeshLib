#pragma once
#include <functional>
#include "MRRibbonMenuItem.h"
#include "imgui.h"

namespace MR
{

struct MenuItemInfo;
class RibbonMenu;
class ShortcutManager;
class RibbonFontManager;
class ImGuiImage;

struct DrawButtonParams
{
    enum class SizeType
    {
        Big, // button is big, underline text is divided to 2 lines (maximum) to fit width 
        SmallText, // button is small, text is on same line
        Small // button is small, no text present
    } sizeType{ SizeType::Big }; // type of button to draw

    ImVec2 itemSize; // size of whole item group, needed for all, this should respect system scaling
    float iconSize{ 0.f }; // icon size ( = 0 - use preset according to button type), this is scale factor of cBigIcon size (should not respect system scaling)
    enum RootType
    {
        Ribbon, // button is on main ribbon bar
        Toolbar, // button is on toolbar
        Header // button is on header quick access bar
    } rootType{ RootType::Ribbon };
};

/// class for drawing ribbon menu buttons
class MRVIEWER_CLASS RibbonButtonDrawer
{
public:
    // Creates GL texture for gradient UI (called on theme apply)
    MRVIEWER_API static void InitGradientTexture();
    MRVIEWER_API static std::unique_ptr<ImGuiImage>& GetGradientTexture();

    /// draw gradient button
    MRVIEWER_API static bool GradientButton( const char* label, const ImVec2& size = ImVec2( 0, 0 ) );
    /// draw gradient button, which can be disabled (valid = false)
    MRVIEWER_API static bool GradientButtonValid( const char* label, bool valid, const ImVec2& size = ImVec2( 0, 0 ) );
    /// draw gradient checkbox
    MRVIEWER_API static bool GradientCheckbox( const char* label, bool* value );
    /// draw gradient checkbox
    template<typename Getter, typename Setter>
    static bool GradientCheckbox( const char* label, Getter get, Setter set )
    {
        bool value = get();
        bool ret = GradientCheckbox( label, &value );
        set( value );
        return ret;
    }
    /// draw gradient radio button
    MRVIEWER_API static bool GradientRadioButton( const char* label, int* v, int valButton );

    MRVIEWER_API static bool CustomCombo( const char* label, int* v, const std::vector<std::string>& options, bool showPreview = true );

    struct ButtonItemWidth
    {
        float baseWidth{ 0.0f };
        float additionalWidth{ 0.0f }; // for small drop buttons
    };
    MRVIEWER_API ButtonItemWidth calcItemWidth( const MenuItemInfo& item, DrawButtonParams::SizeType sizeType );

    /// draw item button
    MRVIEWER_API void drawButtonItem( const MenuItemInfo& item, const DrawButtonParams& params );

    /// draw custom styled button
    MRVIEWER_API bool drawCustomStyledButton( const char* icon, const ImVec2& size, float iconSize );

    /// set reaction on press item button
    void setOnPressAction( std::function<void( std::shared_ptr<RibbonMenuItem>, bool )> action ) { onPressAction_ = action; };
    /// set function to get requirements for activate item
    void setGetterRequirements( std::function<std::string( std::shared_ptr<RibbonMenuItem> )> getterRequirements ) { getRequirements_ = getterRequirements; };

    void setMenu( RibbonMenu* menu ) { menu_ = menu; };
    void setFontMenager( const RibbonFontManager* fontManager ) { fontManager_ = fontManager; };
    void setShortcutManager( const ShortcutManager* shortcutManager ) { shortcutManager_ = shortcutManager; };
    void setScaling( float scaling ) { scaling_ = scaling; };

private:
    void drawButtonDropItem_( const MenuItemInfo& item, const DrawButtonParams& params, bool enabled );
    void drawDropList_( const std::shared_ptr<RibbonMenuItem>& baseDropItem );
    void drawTooltip_( const MenuItemInfo& item, const std::string& requirements );

    // returns num of pushed colors
    int pushRibbonButtonColors_( bool enabled, bool active, DrawButtonParams::RootType rootType ) const;

    std::function<void( std::shared_ptr<RibbonMenuItem>, bool )> onPressAction_ = []( std::shared_ptr<RibbonMenuItem>, bool ) {};
    std::function<std::string( std::shared_ptr<RibbonMenuItem> )> getRequirements_ = []( std::shared_ptr<RibbonMenuItem> ) { return std::string(); };
    RibbonMenu* menu_ = nullptr;
    const RibbonFontManager* fontManager_ = nullptr;
    const ShortcutManager* shortcutManager_ = nullptr;
    float scaling_ = 1.f;
};

}
