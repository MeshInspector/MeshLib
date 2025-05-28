#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRVector2.h"
#include "MRRibbonSchema.h"
#include "MRMesh/MRSignal.h"
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
    const char* windowName() const { return "##RibbonGlobalSearchPopup[rect_allocator_ignore]"; }
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

    // set draw mode (true - small button, false - input string )
    void setSmallUI( bool on ) { isSmallUI_ = on; }

    // get width ui element in top panel
    // return unscaled value 
    MRVIEWER_API float getWidthMenuUI() const;

    // get search string width (+ item spacing)
    MRVIEWER_API float getSearchStringWidth() const;

    // activate search from outside (i.e. shortcut)
    MRVIEWER_API void activate();

    /// set function to get a requirements line for some tool
    void setRequirementsFunc( const RequirementsFunction& requirementsFunc )
    { requirementsFunc_ = requirementsFunc; }

    // this signal is emitted when search bar is focused
    Signal<void()> onFocusSignal;
    // this signal is emitted when tool is activated within search
    Signal<void( std::shared_ptr<RibbonMenuItem> )> onToolActivateSignal;
private:
    bool smallSearchButton_( const Parameters& params );

    void drawWindow_( const Parameters& params );

    void deactivateSearch_();

    bool searchInputText_( const char* label, std::string& str, const RibbonMenuSearch::Parameters& params );

    void updateSearchResult_();

    std::string searchLine_;
    std::vector<RibbonSchemaHolder::SearchResult> searchResult_;
    std::vector<RibbonSchemaHolder::SearchResultWeight> searchResultWeight_;
    std::vector<RibbonSchemaHolder::SearchResult> recentItems_;
    int hightlightedSearchItem_{ 0 };
    int captionCount_ = 0;


    bool isSmallUI_ = false;
    bool active_ = false;
    bool prevFrameActive_ = false;
    bool isSmallUILast_ = false;
    bool mainInputFocused_ = false;
    bool blockSearchBtn_ = false;
    bool setInputFocus_ = false;
    RequirementsFunction requirementsFunc_;
#ifndef NDEBUG
    bool showResultWeight_ = false;
#endif
};

}