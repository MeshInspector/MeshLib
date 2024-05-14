#pragma once
#include "exports.h"

namespace MR
{

class Object;

class MRVIEWER_CLASS SceneObjectsListDrawer
{
public:
    MRVIEWER_API SceneObjectsListDrawer();
    MRVIEWER_API virtual ~SceneObjectsListDrawer() = default;

    MRVIEWER_API void draw( float height, float scaling );

    /// returns flag show detailed information in the object tree
    bool getShowInfoInObjectTree() const { return showInfoInObjectTree_; }
    /// set flag show detailed information in the object tree
    void setShowInfoInObjectTree( bool value ) { showInfoInObjectTree_ = value; }
    
    void setShowNewSelectedObjects( bool show ) { showNewSelectedObjects_ = show; };
    // get show selected objects state (enable / disable)
    bool getShowNewSelectedObjects() { return showNewSelectedObjects_; };
    
    void setDeselectNewHiddenObjects( bool deselect ) { deselectNewHiddenObjects_ = deselect; }
    // get deselect hidden objects state (enable / disable)
    bool getDeselectNewHiddenObjects() { return deselectNewHiddenObjects_; }

    MRVIEWER_API void setObjectTreeState( const Object* obj, bool open );

    MRVIEWER_API void allowSceneReorder( bool allow );

    MRVIEWER_API void setNextFrameFixScroll() { nextFrameFixScroll_ = true; }
protected:
    // override this to customize prefix for objects in scene
    // \detail height should be less or equal ImGui::GetFrameHeight()
    // method should save ImGui::CursorPosY
    MRVIEWER_API virtual void drawCustomObjectPrefixInScene_( const Object& )
    {}
    // override this add custom context menu for selected objects
    MRVIEWER_API virtual void drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& /*selected*/ )
    {}

    // override this to have custom UI in "Scene" window (under opened(expanded) object line)
    // \detail if onlyHeight is true, should return drawing height without rendering
    // return 0.f if nothing drawing
    MRVIEWER_API virtual float drawCustomTreeObjectProperties_( Object& obj, bool onlyCalcHeight );

private:
    void drawObjectsList_();

    // payload object will be moved
    MRVIEWER_API void makeDragDropSource_( const std::vector<std::shared_ptr<Object>>& payload );
    // checking the need to draw a target
    MRVIEWER_API bool needDragDropTarget_();
    // "target" and "before" are "to" and "before" of SceneReorder struct
    // betweenLine - if true requires to draw line (between two objects in tree, for ImGui to have target)
    // counter - unique number of object in tree (needed for ImGui to differ new lines)
    MRVIEWER_API void makeDragDropTarget_( Object& target, bool before, bool betweenLine, const std::string& uniqueStr );
    MRVIEWER_API void reorderSceneIfNeeded_();

    // this function should be called after BeginChild("SceneObjectsList") (child window with scene tree)
    MRVIEWER_API virtual void updateSceneWindowScrollIfNeeded_();

    std::vector<Object*> getPreSelection_( Object* meshclicked,
                                           bool isShift, bool isCtrl,
                                           const std::vector<std::shared_ptr<Object>>& selected,
                                           const std::vector<std::shared_ptr<Object>>& all );


    float menuScaling_ = 1.f;

    bool showInfoInObjectTree_ = false;
    bool showNewSelectedObjects_ = true;
    bool deselectNewHiddenObjects_ = false;

    bool dragTrigger_ = false;
    bool clickTrigger_ = false;
    bool allowSceneReorder_ = true;

    struct SceneReorder
    {
        std::vector<Object*> who; // object that will be moved
        Object* to{ nullptr }; // address object
        bool before{ false }; // if false "who" will be attached to "to" as last child, otherwise "who" will be attached to "to"'s parent as child before "to"
    } sceneReorderCommand_;
    // Drag objects servant data
    // struct to handle changed scene window size scroll
    struct ScrollPositionPreservation
    {
        float relativeMousePos{ 0.0f };
        float absLinePosRatio{ 0.0f };
    } prevScrollInfo_;
    // true to fix scroll position in next frame
    bool nextFrameFixScroll_{ false };
    // flag to know if we are dragging objects now or not
    bool dragObjectsMode_{ false };

    std::unordered_map<const Object*, bool> sceneOpenCommands_;
};

}
