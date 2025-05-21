#pragma once
#include "MRRibbonMenuItem.h"
#include "MRRibbonIcons.h"
#include "MRImGui.h"
#include <functional>

namespace MR
{

struct MenuItemInfo;
class RibbonMenu;
class ShortcutManager;
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

    // if true treat this item as hovered
    bool forceHovered = false;
    // if true treat this item as pressed
    bool forcePressed = false;
    // set true if item pressed
    bool* isPressed = nullptr;
};

struct CustomButtonParameters
{
    // if not set push default ribbon colors
    std::function<int( bool, bool )> pushColorsCb;
    RibbonIcons::IconType iconType;
};

/// class for drawing ribbon menu buttons
class MRVIEWER_CLASS RibbonButtonDrawer
{
public:
    // Creates GL texture for gradient UI (called on theme apply)
    MRVIEWER_API static void InitGradientTexture();
    enum class TextureType
    {
        Mono,
        Gradient,
        RainbowRect,
        Count
    };
    MRVIEWER_API static std::unique_ptr<ImGuiImage>& GetTexture( TextureType type );


    /// draw gradient checkbox with icon (for menu item)
    MRVIEWER_API bool GradientCheckboxItem( const MenuItemInfo& item, bool* value ) const;


    /// draw custom collapsing header
    /// if issueCount is greater than zero, so many red dots will be displayed after text
    MRVIEWER_API static bool CustomCollapsingHeader( const char* label, ImGuiTreeNodeFlags flags = 0, int issueCount = 0 );

    struct ButtonItemWidth
    {
        float baseWidth{ 0.0f };
        float additionalWidth{ 0.0f }; // for small drop buttons
    };
    MRVIEWER_API ButtonItemWidth calcItemWidth( const MenuItemInfo& item, DrawButtonParams::SizeType sizeType ) const;
    
    /// draw item button
    MRVIEWER_API void drawButtonItem( const MenuItemInfo& item, const DrawButtonParams& params ) const;

    /// draw item button
    MRVIEWER_API void drawCustomButtonItem( const MenuItemInfo& item, const CustomButtonParameters& customParam, 
        const DrawButtonParams& params ) const;

    /// draw item button icon
    MRVIEWER_API void drawButtonIcon( const MenuItemInfo& item, const DrawButtonParams& params ) const;

    /// draw custom styled button
    MRVIEWER_API bool drawTabArrowButton( const char* icon, const ImVec2& size, float iconSize );

    /// if set color then instead of multicolored icons will be drawn with this color
    MRVIEWER_API void setMonochrome( const std::optional<Color>& color );

    /// set reaction on press item button
    void setOnPressAction( std::function<void( std::shared_ptr<RibbonMenuItem>, const std::string& )> action ) { onPressAction_ = action; };
    /// set function to get requirements for activate item
    void setGetterRequirements( std::function<std::string( std::shared_ptr<RibbonMenuItem> )> getterRequirements ) { getRequirements_ = getterRequirements; };

    void setMenu( RibbonMenu* menu ) { menu_ = menu; };
    void setShortcutManager( const ShortcutManager* shortcutManager ) { shortcutManager_ = shortcutManager; };
    void setScaling( float scaling ) { scaling_ = scaling; };

    /// returns num of pushed colors
    /// requires to pop it afterwards
    MRVIEWER_API int pushRibbonButtonColors( bool enabled, bool active, bool forceHovered, DrawButtonParams::RootType rootType ) const;

private:
    void drawButtonDropItem_( const MenuItemInfo& item, const DrawButtonParams& params ) const;
    void drawDropList_( const std::shared_ptr<RibbonMenuItem>& baseDropItem ) const;
    void drawTooltip_( const MenuItemInfo& item, const std::string& requirements ) const;

    std::function<void( std::shared_ptr<RibbonMenuItem>, const std::string& )> onPressAction_ = []( std::shared_ptr<RibbonMenuItem>, const std::string& ) {};
    std::function<std::string( std::shared_ptr<RibbonMenuItem> )> getRequirements_ = []( std::shared_ptr<RibbonMenuItem> ) { return std::string(); };
    RibbonMenu* menu_ = nullptr;
    const ShortcutManager* shortcutManager_ = nullptr;

    std::optional<Color> monochrome_;

    float scaling_ = 1.f;
    static std::vector<std::unique_ptr<MR::ImGuiImage>> textures_;
};

}
