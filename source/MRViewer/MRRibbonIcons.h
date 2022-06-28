#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRphmap.h"
#include <array>

namespace MR
{

// this class holds icons for ribbon items
class MRVIEWER_CLASS RibbonIcons
{
public:
    enum class ColorType
    {
        Colored,
        White
    };
    enum class IconType
    {
        RibbonItemIcon,
        ObjectTypeIcon
    };
    // this should be called once on start of programm (called in RibbonMenu::init)
    MRVIEWER_API static void load();
    // this should be called once before programm stops (called in RibbonMenu::shutdown)
    MRVIEWER_API static void free();
    // finds icon with best fitting size, if there is no returns nullptr
    MRVIEWER_API static const ImGuiImage* findByName( const std::string& name, float width, 
                                                      ColorType colorType, IconType iconType );
private:
    RibbonIcons() = default;
    ~RibbonIcons() = default;

    static RibbonIcons& instance_();

    struct Icons
    {
        std::unique_ptr<ImGuiImage> colored;
        std::unique_ptr<ImGuiImage> white;
    };
    enum class Sizes
    {
        X0_5,
        X1,
        X3,
        Count,

        MinRibbonItemIconSize = X0_5,
        MinObjectTypeIconSize = X1,

        MaxRibbonItemIconSize = X3,
        MaxObjectTypeIconSize = X3
    };
    std::array<int, size_t( Sizes::Count )> loadedRibbonItemIconSizes_ = { 0,0,0 };
    std::array<int, size_t( Sizes::Count )> loadedObjectTypeIconSizes_ = { 0,0,0 };

    using SizedIcons = std::array<Icons, size_t( Sizes::Count )>;

    static const char* sizeSubFolder_( Sizes sz );

    Sizes findRequiredSize_( float width, IconType iconType ) const;

    void load_( IconType type );

    HashMap<std::string, SizedIcons> ribbonItemIconsMap_;
    HashMap<std::string, SizedIcons> objectTypeIconsMap_;
};

}