#pragma once

#include "exports.h"
#include "MRSceneReorder.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace MR
{

class Object;

/// class for drawing a list of scene objects (and handling interaction with it)
class MRVIEWER_CLASS SceneObjectsListDrawer
{
public:
    virtual ~SceneObjectsListDrawer() = default;

    /// Main method for drawing all
    /// \detail Not creat window. Use in window block (between ImGui::Begin and ImGui::End)
    MRVIEWER_API virtual void draw( float height, float scaling );
    
    /// set flag of the object visibility activation after selection
    void setShowNewSelectedObjects( bool show ) { showNewSelectedObjects_ = show; };
    /// get flag of the object visibility activation after selection
    bool getShowNewSelectedObjects() { return showNewSelectedObjects_; };
    
    /// set flag of deselect object after hidden
    void setDeselectNewHiddenObjects( bool deselect ) { deselectNewHiddenObjects_ = deselect; }
    /// get flag of deselect object after hidden
    bool getDeselectNewHiddenObjects() { return deselectNewHiddenObjects_; }

    /// change selection after pressed arrow up / down
    /// isDown - true if pressed arrow down, false - arrow up
    /// isShift - shift button holded
    MRVIEWER_API void changeSelection( bool isDown, bool isShift );
    /// change visible (and selection) after pressed arrow up / down
    /// isDown - true if pressed F4, false - F3
    MRVIEWER_API void changeVisible( bool isDown );

    // select all selectable objects
    MRVIEWER_API void selectAllObjects();

    // stt visible flag all selectable objects
    MRVIEWER_API void setLeavesVisibility( bool visible );

    /// set object collapse state (hiding children)
    MRVIEWER_API void setObjectTreeState( const Object* obj, bool open );

    /// set possibility change object order
    MRVIEWER_API void allowSceneReorder( bool allow );

    /// helper method for fix scroll position after change available height
    MRVIEWER_API void setNextFrameFixScroll() { nextFrameFixScroll_ = true; }
protected:
    /// override this to customize prefix for objects in scene
    /// \detail height should be less or equal ImGui::GetFrameHeight()
    /// method should save ImGui::CursorPosY
    MRVIEWER_API virtual void drawCustomObjectPrefixInScene_( const Object&, bool )
    {}
    /// override this add custom context menu for selected objects
    /// uniqueStr need to identify who call context menu
    MRVIEWER_API virtual void drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& /*selected*/, const std::string& /*uniqueStr*/ )
    {}

    /// override this to have custom UI in "Scene" window (under opened(expanded) object line)
    /// \detail if onlyHeight is true, should return drawing height without rendering
    /// return 0.f if nothing drawing
    MRVIEWER_API virtual float drawCustomTreeObjectProperties_( Object& obj, bool onlyCalcHeight );

    typedef int ImGuiTreeNodeFlags;
    /// override this to customize CollapsingHeader draw
    MRVIEWER_API virtual bool collapsingHeader_( const std::string& uniqueName, ImGuiTreeNodeFlags flags );

    MRVIEWER_API virtual std::string objectLineStrId_( const Object& object, const std::string& uniqueStr );

    /// override this to customize whole object line
    MRVIEWER_API virtual bool drawObject_( Object& object, const std::string& uniqueStr, int depth );
    /// draw dummy container for skipped object
    MRVIEWER_API virtual bool drawSkippedObject_( Object& object, const std::string& uniqueStr, int depth );

    /// function that draws collapsing header and process click on it
    MRVIEWER_API bool drawObjectCollapsingHeader_( Object& object, const std::string& uniqueStr, bool hasRealChildren );

    /// function that do click logic on object line (select/deselect/rename/open context)
    MRVIEWER_API void processItemClick_( Object& object, const std::vector<std::shared_ptr<Object>>& selected );

    /// payload object will be moved
    MRVIEWER_API void makeDragDropSource_( const std::vector<std::shared_ptr<Object>>& payload );

    /// checking the need to draw a target
    MRVIEWER_API bool needDragDropTarget_();

    /// "target" and "before" are "to" and "before" of SceneReorder struct
    /// betweenLine - if true requires to draw line (between two objects in tree, for ImGui to have target)
    /// counter - unique number of object in tree (needed for ImGui to differ new lines)
    MRVIEWER_API void makeDragDropTarget_( Object& target, bool before, bool betweenLine, const std::string& uniqueStr );

    // draw visibility checkbox and call ImGui::SameLine at the end
    MRVIEWER_API void drawObjectVisibilityCheckbox_( Object& object, const std::string& uniqueStr );

    bool showNewSelectedObjects_ = true;
    bool deselectNewHiddenObjects_ = false;
private:
    void drawObjectsList_();

    float getDrawDropTargetHeight_() const { return 4.f * menuScaling_; }
    void reorderSceneIfNeeded_();

    /// this function should be called after BeginChild("SceneObjectsList") (child window with scene tree)
    MRVIEWER_API virtual void updateSceneWindowScrollIfNeeded_();

    std::vector<Object*> getPreSelection_( Object* meshclicked,
                                           bool isShift, bool isCtrl,
                                           const std::vector<std::shared_ptr<Object>>& selected,
                                           const std::vector<std::shared_ptr<Object>>& all );
    void updateSelection_( Object* objPtr, const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all );

    bool dragTrigger_ = false;
    bool clickTrigger_ = false;
    bool allowSceneReorder_ = true;

    // struct to auto-scroll after move (arrow up / down)
    struct MoveAndScrollData
    {
        int index = -1; // index of new selected object in list of all
        float posY = -1.f; // scroll position of new selected object in list of all
        bool needScroll = false; // flag to using auto-scroll
    };
    MoveAndScrollData upFirstSelected_;
    MoveAndScrollData downLastSelected_;
    MoveAndScrollData nextVisible_;

    SceneReorder sceneReorderCommand_;
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

protected:
    float menuScaling_ = 1.f;
    std::unordered_map<const Object*, bool> sceneOpenCommands_;
};

}
