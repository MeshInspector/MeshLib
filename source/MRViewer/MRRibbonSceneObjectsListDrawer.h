#pragma once
#include "MRSceneObjectsListDrawer.h"

namespace MR
{

class RibbonMenu;

class MRVIEWER_CLASS RibbonSceneObjectsListDrawer : public SceneObjectsListDrawer
{
public:
    void initRibbonMenu( RibbonMenu* ribbonMenu ) { ribbonMenu_ = ribbonMenu; };
    
    // for enable / disable close scene context menu on any change
    void setCloseContextOnChange( bool deselect ) { closeContextOnChange_ = deselect; }
    bool getCloseContextOnChange() { return closeContextOnChange_; }

protected:
    MRVIEWER_API virtual void drawCustomObjectPrefixInScene_( const Object& obj ) override;
    MRVIEWER_API virtual void drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& selected ) override;

private:
    // return icon (now it is symbol in icons font) based on typename
    MRVIEWER_API virtual const char* getSceneItemIconByTypeName_( const std::string& typeName ) const;

    RibbonMenu* ribbonMenu_ = nullptr;
    bool closeContextOnChange_ = true;
};

}
