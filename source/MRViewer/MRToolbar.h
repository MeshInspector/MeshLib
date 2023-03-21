#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRColor.h"
#include "json/value.h"

namespace MR
{
using MenuItemsList = std::vector<std::string>;

class RibbonMenu;

/// class to draw toolbar and toolbar customize windows
class Toolbar
{
public:
    /// set pointer on ribbon menu to access it
    void setRibbonMenu( RibbonMenu* ribbonMenu );

    /// draw toolbar window
    /// \details don't show if there isn't any items or not enough space
    void drawToolbar();

    // enable toolbar customize window rendering
    void openCustomize();
    /// draw toolbar customize window
    /// \details window is modal window
    void drawCustomize();

    /// read toolbar items from json
    void readItemsList( const Json::Value& root );
    /// reset items list to default value
    /// \detail default value is taken from RibbonSchemaHolder
    void resetItemsList();
    /// get acces to items
    const MenuItemsList& getItemsList() const { return itemsList_; }
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

    bool dragDrop_ = false; // active drag&drop in Toolbar Customize window
    bool openCustomizeFlag_ = false; // flag to open Toolbar Customize window
    int customizeTabNum_ = 0; // number active tab
    std::string searchString_;
    std::vector<std::vector<std::string>> searchResult_;
};

}
