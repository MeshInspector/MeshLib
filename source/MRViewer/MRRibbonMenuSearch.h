#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRVector2.h"
#include "MRRibbonSchema.h"
#include <string>
#include <functional>

namespace MR
{

class RibbonButtonDrawer;
class RibbonFontManager;

// separate class for search in ribbon menu
class MRVIEWER_CLASS RibbonMenuSearch
{
public:
    // returns search imgui popup window name
    const char* windowName() const { return "##RibbonGlobalSearchPopup"; }
    // add item to recent items list
    void pushRecentItem( const std::shared_ptr<RibbonMenuItem>& item );

    struct Parameters
    {
        RibbonButtonDrawer& btnDrawer;
        RibbonFontManager& fontManager;
        std::function<void( int )> changeTabFunc;
        float scaling;
    };
    // draws search elements and window with its logic
    MRVIEWER_API void drawMenuUI( const Parameters& params );

    // get mode visualization in top panel (true - small button, false - input string) 
    MRVIEWER_API bool isSmallUI() const;

    // get width ui element in top panel
    // return unscaled value 
    MRVIEWER_API float getWidthMenuUI() const;

    // activate search from outside (i.e. shortcut)
    MRVIEWER_API void activate();
private:
    bool smallSearchButton_( const Parameters& params );

    void drawWindow_( const Parameters& params );

    void deactivateSearch_();

    std::string searchLine_;
    std::vector<RibbonSchemaHolder::SearchResult> searchResult_;
    std::vector<RibbonSchemaHolder::SearchResult> recentItems_;
    int hightlightedSearchItem_{ -1 };

    bool active_ = false;
    bool isSmallUILast_ = false;
    bool mainInputFocused_ = false;
    bool blockSearchBtn_ = false;
    bool setMainInputFocus_ = false;
};

}