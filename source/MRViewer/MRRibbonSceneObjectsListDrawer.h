#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRSceneObjectsListDrawer.h"
#include "MRMesh/MRSignal.h"

namespace MR
{

class RibbonMenu;

/// class for drawing a list of scene objects in RibbonMenu style
class MRVIEWER_CLASS RibbonSceneObjectsListDrawer : public SceneObjectsListDrawer
{
public:
    MRVIEWER_API virtual void draw( float height, float scaling ) override;

    MRVIEWER_API void initRibbonMenu( RibbonMenu* ribbonMenu );
    
    /// set closing scene context menu on any change
    void setCloseContextOnChange( bool deselect ) { closeContextOnChange_ = deselect; }
    /// get flag closing scene context menu on any change
    bool getCloseContextOnChange() { return closeContextOnChange_; }

    /// this signal is emitted each frame inside scene context window
    Signal<void()> onDrawContextSignal;
protected:
    MRVIEWER_API virtual void drawCustomObjectPrefixInScene_( const Object& obj, bool opened ) override;
    MRVIEWER_API virtual void drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& selected, const std::string& uniqueStr ) override;
    MRVIEWER_API virtual bool collapsingHeader_( const std::string& uniqueName, ImGuiTreeNodeFlags flags ) override;

    MRVIEWER_API virtual std::string objectLineStrId_( const Object& object, const std::string& uniqueStr ) override;

    MRVIEWER_API virtual bool drawObject_( Object& object, const std::string& uniqueStr, int depth ) override;
    MRVIEWER_API virtual bool drawSkippedObject_( Object& object, const std::string& uniqueStr, int depth ) override;
private:
    // return icon (now it is symbol in icons font) based on typename
    MRVIEWER_API virtual const char* getSceneItemIconByTypeName_( const std::string& typeName ) const;

    bool drawTreeOpenedState_( Object& object, bool leaf, const std::string& uniqueStr, int depth );
    void drawObjectLine_( Object& object, const std::string& uniqueStr, bool opened );
    void drawEyeButton_( Object& object, const std::string& uniqueStr, bool frameHovered );

    void drawHierarhyLine_( const Vector2f& startScreenPos, int depth, bool skipped );
    
    struct LastDepthInfo
    {
        float screenPosY{ 0.0f };
        int id{ 0 };
    };
    // depth -> pos Y of last element of this depth
    std::vector<LastDepthInfo> lastDrawnSibling_;
    int currentElementId_{ 0 };
    RibbonMenu* ribbonMenu_ = nullptr;
    bool closeContextOnChange_ = true;
};

}
