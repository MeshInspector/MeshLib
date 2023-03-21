#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRColor.h"

namespace MR
{
using MenuItemsList = std::vector<std::string>;

class RibbonMenu;

class Toolbar
{
public:
    void setRibbonMenu( RibbonMenu* ribbonMenu );

    void drawToolbar();

    void openCustomize();
    void drawCustomize();

    void readItemsList( const Json::Value& root );
    void resetItemsList();
    const MenuItemsList& getItemsList() const { return itemsList_; }
private:
    void drawCustomizeModal_();
    void drawCustomizeTabsList_();
    void drawCustomizeItemsList_();

    void dashedLine_( const Vector2f& org, const Vector2f& dest, float periodLength = 10.f, float fillRatio = 0.5f, const Color& color = Color::gray(), float periodStart = 0.f );
    void dashedRect_( const Vector2f& leftTop, const Vector2f& rightBottom, float periodLength = 10.f, float fillRatio = 0.5f, const Color& color = Color::gray() );

    RibbonMenu* ribbonMenu_;

    float scaling_ = 1.f;

    MenuItemsList itemsList_; // toolbar items list
    MenuItemsList itemsListCustomize_; // toolbar preview items list for Toolbar Customize window

    bool dragDrop_ = false; // active drag&drop in Toolbar Customize window
    bool openCustomizeFlag_ = false; // flag to open Toolbar Customize window
    int customizeTabNum_ = 0;
    std::string searchString_;
    std::vector<std::vector<std::string>> searchResult_;
};

}
