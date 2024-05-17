#pragma once
#include "MRSceneObjectsListDrawer.h"

namespace MR
{

class RibbonMenu;

/// class for drawing a list of scene objects in RibbonMenu style
class MRVIEWER_CLASS RibbonSceneObjectsListDrawer : public SceneObjectsListDrawer
{
public:
    void initRibbonMenu( RibbonMenu* ribbonMenu ) { ribbonMenu_ = ribbonMenu; };
    
    /// set closing scene context menu on any change
    void setCloseContextOnChange( bool deselect ) { closeContextOnChange_ = deselect; }
    /// get flag closing scene context menu on any change
    bool getCloseContextOnChange() { return closeContextOnChange_; }

protected:
    MRVIEWER_API virtual void drawCustomObjectPrefixInScene_( const Object& obj ) override;
    MRVIEWER_API virtual void drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& selected, const std::string uniqueStr ) override;
    MRVIEWER_API virtual bool collapsingHeader_( const std::string& uniqueName, ImGuiTreeNodeFlags flags ) override;

private:
    // return icon (now it is symbol in icons font) based on typename
    MRVIEWER_API virtual const char* getSceneItemIconByTypeName_( const std::string& typeName ) const;

    RibbonMenu* ribbonMenu_ = nullptr;
    bool closeContextOnChange_ = true;
};

}
