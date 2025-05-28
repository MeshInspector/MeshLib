#pragma once

#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRColor.h"
#include <json/forwards.h>
#include <map>
#include <vector>
#include <string>

namespace MR
{
using MenuItemsList = std::vector<std::string>;
using MenuItemsListMigration = std::function<void ( MenuItemsList& )>;
using MenuItemsListMigrations = std::map<int, MenuItemsListMigration>;

class RibbonMenu;

/// class to draw toolbar and toolbar customize windows
class MRVIEWER_CLASS Toolbar
{
public:
    /// set pointer on ribbon menu to access it
    MRVIEWER_API void setRibbonMenu( RibbonMenu* ribbonMenu );

    /// draw toolbar window
    /// \details don't show if there isn't any items or not enough space
    MRVIEWER_API void drawToolbar();
    /// return current width of toolbar
    /// 0.0 if it is not present
    MRVIEWER_API float getCurrentToolbarWidth() const { return currentWidth_; }
    // enable toolbar customize window rendering
    MRVIEWER_API void openCustomize();
    /// draw toolbar customize window
    /// \details window is modal window
    MRVIEWER_API void drawCustomize();

    /// read toolbar items from json
    MRVIEWER_API void readItemsList( const Json::Value& root );
    /// reset items list to default value
    /// \details default value is taken from RibbonSchemaHolder
    MRVIEWER_API void resetItemsList();
    /// get access to items
    MRVIEWER_API const MenuItemsList& getItemsList() const { return itemsList_; }
    /// get item list version
    MRVIEWER_API int getItemsListVersion() const { return itemsListVersion_; }
    /// set item list version
    MRVIEWER_API void setItemsListVersion( int version ) { itemsListVersion_ = version; }
    /// set item list's upgrade rules
    MRVIEWER_API void setItemsListMigrations( const MenuItemsListMigrations& migrations ) { itemsListMigrations_ = migrations; }

    MRVIEWER_API void setScaling( float scale ) { scaling_ = scale; }

    void setMaxItemCount( int maxItemCount ) { maxItemCount_ = maxItemCount; }
    int getMaxItemCount() const { return maxItemCount_; }

private:
    /// draw toolbar customize modal
    void drawCustomizeModal_();
    /// draw tabs list
    void drawCustomizeTabsList_();
    /// draw items list, search and reset btn
    void drawCustomizeItemsList_();

    void dashedLine_( const Vector2f& org, const Vector2f& dest, float periodLength = 10.f, float fillRatio = 0.5f, const Color& color = Color::gray(), float periodStart = 0.f );
    void dashedRect_( const Vector2f& leftTop, const Vector2f& rightBottom, float periodLength = 10.f, float fillRatio = 0.5f, const Color& color = Color::gray() );

    RibbonMenu* ribbonMenu_;

    float scaling_ = 1.f;

    MenuItemsList itemsList_; // toolbar items list
    MenuItemsList itemsListCustomize_; // toolbar preview items list for Toolbar Customize window
    int itemsListVersion_{ 1 }; // items list version
    MenuItemsListMigrations itemsListMigrations_; // items list's upgrade rules

    float currentWidth_{ 0.0f };
    bool dragDrop_ = false; // active drag&drop in Toolbar Customize window
    bool openCustomizeFlag_ = false; // flag to open Toolbar Customize window
    int customizeTabNum_ = 0; // number active tab
    std::string searchString_;
    std::vector<std::vector<std::string>> searchResult_;

    int maxItemCount_ = 14;
};

}
