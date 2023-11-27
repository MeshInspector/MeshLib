#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRVector2.h"
#include "MRRibbonSchema.h"
#include <string>
#include <functional>

namespace MR
{

class RibbonButtonDrawer;

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
        Vector2f absMinPos;
        std::function<void( int )> changeTabFunc;
        float scaling;
    };
    // draws search popup window with its logic
    MRVIEWER_API void draw( const Parameters& params );
private:
    std::string searchLine_;
    std::vector<RibbonSchemaHolder::SearchResult> searchResult_;
    std::vector<RibbonSchemaHolder::SearchResult> recentItems_;
    int hightlightedSearchItem_{ -1 };
};

}