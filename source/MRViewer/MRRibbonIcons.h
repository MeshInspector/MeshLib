#pragma once
#include "MRMesh/MRFlagOperators.h"
#include "MRMesh/MRphmap.h"
#include "MRViewerFwd.h"
#include <array>
#include <filesystem>

namespace MR
{

// this class holds icons for ribbon items
class MRVIEWER_CLASS RibbonIcons
{
public:
    enum class ColorType
    {
        Colored,
        White,
    };
    enum class IconType
    {
        RibbonItemIcon,   // have four sizes
        ObjectTypeIcon,   // have two sizes
        IndependentIcons, // have two sizes
        Logos,            // have two sizes
        Count,
    };
    // this should be called once on start of program (called in RibbonMenu::init)
    MRVIEWER_API static void load();
    // this should be called once before program stops (called in RibbonMenu::shutdown)
    MRVIEWER_API static void free();
    // finds icon with best fitting size, if there is no returns nullptr
    MRVIEWER_API static const ImGuiImage* findByName( const std::string& name, float width, 
                                                      ColorType colorType, IconType iconType );
private:
    RibbonIcons();
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
        X0_75,
        X1,
        X3,
        Count,
    };

    using SizedIcons = std::array<Icons, size_t( Sizes::Count )>;

    static const char* sizeSubFolder_( Sizes sz );

    const ImGuiImage* findRequiredSize_( const SizedIcons& icons, float width, ColorType colorType, IconType iconType ) const;

    void load_( IconType type );

    struct IconTypeData
    {
        enum class AvailableColor
        {
            White = 1 << 0,
            Colored = 1 << 1,
        };
        MR_MAKE_FLAG_OPERATORS_IN_CLASS( AvailableColor )

        std::filesystem::path pathDirectory;
        // first - min size, second - max size
        std::pair<Sizes, Sizes> minMaxSizes;
        AvailableColor availableColor = AvailableColor::White;
        HashMap<std::string, SizedIcons> map;
    };

    std::array<IconTypeData, size_t( IconType::Count )> data_;
};

}