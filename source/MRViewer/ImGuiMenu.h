#pragma  once
// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2018 Jérémie Dumas <jeremie.dumas@ens-lyon.org>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "MRMesh/MRFlagOperators.h"
#include "MRViewerPlugin.h"
#include "MRViewerEventsListener.h"
#include "MRStatePlugin.h"
#include "MRNotificationType.h"
#include <unordered_map>
#include "MRMesh/MRIRenderObject.h" //only for BasicUiRenderTask::BackwardPassParams
#include "MRMesh/MRBox.h"

// Forward declarations
struct ImGuiContext;
struct ImGuiWindow;

namespace MR
{

class ShortcutManager;
class MeshModifier;
struct UiRenderManager;

enum class SelectedTypesMask
{
    ObjectBit = 1 << 0,
    ObjectPointsHolderBit = 1 << 1,
    ObjectLinesHolderBit = 1 << 2,
    ObjectMeshHolderBit = 1 << 3,
    ObjectLabelBit = 1 << 4,
    ObjectMeshBit = 1 << 5,
    ObjectFeaturesBit = 1 << 6,
};
MR_MAKE_FLAG_OPERATORS( SelectedTypesMask )

class MRVIEWER_CLASS ImGuiMenu : public MR::ViewerPlugin,
    public MultiListener<
    MouseDownListener, MouseMoveListener, MouseUpListener, MouseScrollListener, CursorEntranceListener,
    CharPressedListener, KeyDownListener, KeyUpListener, KeyRepeatListener,
    SpaceMouseMoveListener, SpaceMouseDownListener,
    TouchpadRotateGestureBeginListener, TouchpadRotateGestureUpdateListener, TouchpadRotateGestureEndListener,
    TouchpadSwipeGestureBeginListener, TouchpadSwipeGestureUpdateListener, TouchpadSwipeGestureEndListener,
    TouchpadZoomGestureBeginListener, TouchpadZoomGestureUpdateListener, TouchpadZoomGestureEndListener,
    PostResizeListener, PostRescaleListener>
{
    using ImGuiMenuMultiListener = MultiListener<
        MouseDownListener, MouseMoveListener, MouseUpListener, MouseScrollListener,
        CharPressedListener, KeyDownListener, KeyUpListener, KeyRepeatListener,
        SpaceMouseMoveListener, SpaceMouseDownListener,
        TouchpadRotateGestureBeginListener, TouchpadRotateGestureUpdateListener, TouchpadRotateGestureEndListener,
        TouchpadSwipeGestureBeginListener, TouchpadSwipeGestureUpdateListener, TouchpadSwipeGestureEndListener,
        TouchpadZoomGestureBeginListener, TouchpadZoomGestureUpdateListener, TouchpadZoomGestureEndListener,
        PostResizeListener, PostRescaleListener>;
protected:
  // Hidpi scaling to be used for text rendering.
  float hidpi_scaling_;

  // Ratio between the framebuffer size and the window size.
  // May be different from the hipdi scaling!
  float pixel_ratio_;

  // ImGui Context
  ImGuiContext * context_ = nullptr;
  // last focused plugin window
  ImGuiWindow* prevFrameFocusPlugin_ = nullptr;

  // if true, then pre_draw will start from polling glfw events
  bool pollEventsInPreDraw = false; // be careful here with true, this can cause infinite recurse

  bool showShortcuts_{ false };
  bool showStatistics_{ false };
  long long frameTimeMillisecThreshold_{ 25 };
  bool showRenameModal_{ false };
  std::string renameBuffer_;
  std::string popUpRenameBuffer_;
  bool needModalBgChange_{ false };
  bool showInfoModal_{ false };
  std::string storedModalMessage_;
  NotificationType modalMessageType_{ NotificationType::Error };
  std::shared_ptr<ShortcutManager> shortcutManager_;

  ImVec2 sceneWindowPos_;
  ImVec2 sceneWindowSize_;
  ImVec2 mainWindowPos_;
  ImVec2 mainWindowSize_;

  std::unordered_map<const Object*, bool> sceneOpenCommands_;

  MRVIEWER_API virtual void setupShortcuts_();

  bool allowSceneReorder_{ true };
  bool dragTrigger_ = false;
  bool clickTrigger_ = false;
  bool showNewSelectedObjects_{ true };
  bool deselectNewHiddenObjects_{ false };
  bool savedDialogPositionEnabled_{ false };

  struct SceneReorder
  {
      std::vector<Object*> who; // object that will be moved
      Object* to{ nullptr }; // address object
      bool before{ false }; // if false "who" will be attached to "to" as last child, otherwise "who" will be attached to "to"'s parent as child before "to"
  } sceneReorderCommand_;

  std::weak_ptr<Object> lastRenameObj_;
  Box3f selectionBbox_; // updated in drawSelectionInformation_
  Box3f selectionWorldBox_;

  struct LabelParams
  {
      std::string lastLabel;
      std::string labelBuffer;
      std::shared_ptr<ObjectLabel> obj{ nullptr };
  } oldLabelParams_;

  bool allowRemoval_{ true };
  bool uniformScale_{ true };
  bool xfHistUpdated_{ false };
  bool invertedRotation_{ false };
  bool showInfoInObjectTree_{ false };

  std::optional<std::pair<std::string, Vector4f>> storedColor_;
  Vector4f getStoredColor_( const std::string& str, const Color& defaultColor ) const;

  mutable struct PluginsCache
  {
      // if cache is valid do nothing, otherwise accumulate all custom plugins in tab sections and sort them by special string
      void validate( const std::vector<ViewerPlugin*>& viewerPlugins );
      // finds enabled custom plugin, nullptr if none is
      StateBasePlugin* findEnabled() const;
      const std::vector<StateBasePlugin*>& getTabPlugins( StatePluginTabs tab ) const;
  private:
      std::array<std::vector<StateBasePlugin*>, size_t( StatePluginTabs::Count )> sortedCustomPlufins_;
      std::vector<ViewerPlugin*> allPlugins_; // to validate
  } pluginsCache_;

  std::string searchPluginsString_;

  std::vector<std::shared_ptr<MR::MeshModifier>> modifiers_;

  enum ViewportConfigurations
  {
      Single,
      Horizontal, // left viewport, right viewport
      Vertical, // lower viewport, upper viewport
      Quad // left lower vp, left upper vp, right lower vp, right upper vp
  } viewportConfig_{ Single };

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
  // flag to correctly update scroll on transform window appearing
  bool selectionChangedToSingleObj_{ false };
  // this function should be called after BeginChild("Meshes") (child window with scene tree)
  MRVIEWER_API virtual void updateSceneWindowScrollIfNeeded_();
  // menu will change objects' colors in this viewport
  ViewportId selectedViewport_ = {};

  // When editing feature properties, this is the target object.
  std::weak_ptr<Object> editedFeatureObject_;
  // When editing feature properties, this is the original xf of the target object, for history purposes.
  AffineXf3f editedFeatureObjectOldXf_;

public:
  MRVIEWER_API static const std::shared_ptr<ImGuiMenu>& instance();

  MRVIEWER_API virtual void init(MR::Viewer *_viewer) override;

  // inits glfw and glsl backend
  MRVIEWER_API virtual void initBackend();

  // call this to validate imgui context in the begining of the frame
  MRVIEWER_API virtual void startFrame();
  // call this to draw valid imgui context at the end of the frame
  MRVIEWER_API virtual void finishFrame();

  MRVIEWER_API virtual void load_font(int font_size = 13);
  MRVIEWER_API virtual void reload_font(int font_size = 13);

  MRVIEWER_API virtual void shutdown() override;

  // Draw menu
  MRVIEWER_API virtual void draw_menu();

  MRVIEWER_API void draw_helpers();

  // Can be overwritten by `callback_draw_viewer_window`
  MRVIEWER_API virtual void draw_viewer_window();

  MRVIEWER_API void draw_mr_menu();

  // Can be overwritten by `callback_draw_viewer_menu`
  //virtual void draw_viewer_menu();

  // Can be overwritten by `callback_draw_custom_window`
  virtual void draw_custom_window() {}

  // Easy-to-customize callbacks
  std::function<void(void)> callback_draw_viewer_window;
  std::function<void(void)> callback_draw_viewer_menu;
  std::function<void(void)> callback_draw_custom_window;

  void draw_labels_window();

  void draw_labels( const VisualObject& obj );

  MRVIEWER_API void draw_text(
      const Viewport& viewport,
      const Vector3f& pos,
      const Vector3f& normal,
      const std::string& text,
      const Color& color,
      bool clipByViewport );

  MRVIEWER_API float pixel_ratio();

  MRVIEWER_API float hidpi_scaling();

  MRVIEWER_API float menu_scaling() const;

  MRVIEWER_API ImGuiContext* getCurrentContext() const;

  ImGuiWindow* getLastFocusedPlugin() const { return prevFrameFocusPlugin_; };

  // opens Error / Warning / Info modal window with message text
  MRVIEWER_API void showModalMessage( const std::string& msg, NotificationType msgType );

  MRVIEWER_API virtual std::filesystem::path getMenuFontPath() const;

  // setup maximum good time for frame rendering (if rendering is slower it will become red in statistics window)
  MRVIEWER_API void setDrawTimeMillisecThreshold( long long maxGoodTimeMillisec );

  // Draw scene list window with content
  MRVIEWER_API void draw_scene_list();
  // Draw scene list content only
  MRVIEWER_API void draw_scene_list_content( const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all );

  // override this to have custom "Selection Properties" window
  // draw window with content
  MRVIEWER_API virtual void draw_selection_properties( std::vector<std::shared_ptr<Object>>& selected );
  // override this to have custom "Selection Properties" content
  // draw content only
  MRVIEWER_API virtual void draw_selection_properties_content( std::vector<std::shared_ptr<Object>>& selected );
  // override this to have custom UI in "Selection Properties" window (under "Draw Options")

  // override this to customize prefix for objects in scene
  MRVIEWER_API virtual void drawCustomObjectPrefixInScene_( const Object& )
  {}
  // override this to customize appearance of collapsing headers
  MRVIEWER_API virtual bool drawCollapsingHeader_( const char* label, ImGuiTreeNodeFlags flags = 0);
  // override this to customize appearance of collapsing headers for transform block
  MRVIEWER_API virtual bool drawCollapsingHeaderTransform_();
  // override this to have custom UI in "Scene" window (under opened(expanded) object line)
  MRVIEWER_API virtual void draw_custom_tree_object_properties( Object& obj );

  bool make_visualize_checkbox( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label, AnyVisualizeMaskEnum type, MR::ViewportMask viewportid, bool invert = false );
  template<typename ObjectT>
  void make_color_selector( std::vector<std::shared_ptr<ObjectT>> selectedVisualObjs, const char* label,
                            std::function<Vector4f( const ObjectT* )> getter,
                            std::function<void( ObjectT*, const Vector4f& )> setter );
  template<typename ObjType>
  void make_width( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label,
                   std::function<float( const ObjType* )> getter,
                   std::function<void( ObjType*, const float& )> setter,
                   bool lineWidth = false );

  void make_light_strength( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label,
    std::function<float( const VisualObject* )> getter,
    std::function<void( VisualObject*, const float& )> setter);

  template <typename T, typename ObjectType>
  void make_slider( std::vector<std::shared_ptr<ObjectType>> selectedVisualObjs, const char* label,
    std::function<T( const ObjectType* )> getter,
    std::function<void( ObjectType*, T )> setter, T min, T max );

  void make_points_discretization( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label,
  std::function<int( const ObjectPointsHolder* )> getter,
  std::function<void( ObjectPointsHolder*, const int& )> setter );

  MRVIEWER_API void draw_custom_plugins();

  void setShowNewSelectedObjects( bool show ) { showNewSelectedObjects_ = show; };
  // get show selected objects state (enable / disable)
  bool getShowNewSelectedObjects() { return showNewSelectedObjects_; };
  void setDeselectNewHiddenObjects( bool deselect ) { deselectNewHiddenObjects_ = deselect; }
  // get deselect hidden objects state (enable / disable)
  bool getDeselectNewHiddenObjects() { return deselectNewHiddenObjects_; }

  std::shared_ptr<ShortcutManager> getShortcutManager() { return shortcutManager_; };

  MRVIEWER_API void add_modifier( std::shared_ptr<MR::MeshModifier> modifier );

  MRVIEWER_API void allowSceneReorder( bool allow );
  bool checkPossibilityObjectRemoval() { return allowRemoval_; };

  MRVIEWER_API void allowObjectsRemoval( bool allow );

  MRVIEWER_API void tryRenameSelectedObject();

  MRVIEWER_API void setObjectTreeState( const Object* obj, bool open );

  //set show shortcuts state (enable / disable)
  MRVIEWER_API void setShowShortcuts( bool val );
  //return show shortcuts state (enable / disable)
  MRVIEWER_API bool getShowShortcuts() const;

  // enables using of saved positions of plugin windows in the config file
  void enableSavedDialogPositions( bool on ) { savedDialogPositionEnabled_ = on; }
  // returns true if enabled using of saved positions of plugin windows in the config file, false otherwise
  bool isSavedDialogPositionsEnabled() const { return savedDialogPositionEnabled_; }

  // This class helps the viewer to `renderUi()` from `IRenderObject`s.
  MRVIEWER_API virtual UiRenderManager& getUiRenderManager();  

  /// returns flag show detailed information in the object tree
  bool getShowInfoInObjectTree() const { return showInfoInObjectTree_; }
  /// set flag show detailed information in the object tree
  void setShowInfoInObjectTree( bool value ) { showInfoInObjectTree_ = value; }

protected:
    MRVIEWER_API virtual void drawModalMessage_();

    bool capturedMouse_{ false };
    // Mouse IO
    MRVIEWER_API virtual bool onMouseDown_( MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;
    MRVIEWER_API virtual bool onMouseScroll_( float delta_y ) override;
    MRVIEWER_API virtual void cursorEntrance_( bool entered ) override;
    // Keyboard IO
    MRVIEWER_API virtual bool onCharPressed_( unsigned key, int modifiers ) override;
    MRVIEWER_API virtual bool onKeyDown_( int key, int modifiers ) override;
    MRVIEWER_API virtual bool onKeyUp_( int key, int modifiers ) override;
    MRVIEWER_API virtual bool onKeyRepeat_( int key, int modifiers ) override;
    // Scene events
    MRVIEWER_API virtual void postResize_( int width, int height ) override;
    MRVIEWER_API virtual void postRescale_( float x, float y) override;
    // Spacemouse events
    MRVIEWER_API virtual bool spaceMouseMove_( const Vector3f& translate, const Vector3f& rotate ) override;
    MRVIEWER_API virtual bool spaceMouseDown_( int key ) override;
    // Touchpad gesture events
    MRVIEWER_API virtual bool touchpadRotateGestureBegin_() override;
    MRVIEWER_API virtual bool touchpadRotateGestureUpdate_( float angle ) override;
    MRVIEWER_API virtual bool touchpadRotateGestureEnd_() override;
    MRVIEWER_API virtual bool touchpadSwipeGestureBegin_() override;
    MRVIEWER_API virtual bool touchpadSwipeGestureUpdate_( float deltaX, float deltaY, bool kinetic ) override;
    MRVIEWER_API virtual bool touchpadSwipeGestureEnd_() override;
    MRVIEWER_API virtual bool touchpadZoomGestureBegin_() override;
    MRVIEWER_API virtual bool touchpadZoomGestureUpdate_( float scale, bool kinetic ) override;
    MRVIEWER_API virtual bool touchpadZoomGestureEnd_() override;

    // This function reset ImGui style to current theme and scale it by menu_scaling
    // called in ImGuiMenu::postRescale_()
    MRVIEWER_API virtual void rescaleStyle_();

    MRVIEWER_API virtual void addMenuFontRanges_( ImFontGlyphRangesBuilder& builder ) const;

    // payload object will be moved
    MRVIEWER_API void makeDragDropSource_( const std::vector<std::shared_ptr<Object>>& payload );
    // "target" and "before" are "to" and "before" of SceneReorder struct
    // betweenLine - if true requires to draw line (between two objects in tree, for ImGui to have target)
    // counter - unique number of object in tree (needed for ImGui to differ new lines)
    MRVIEWER_API void makeDragDropTarget_( Object& target, bool before, bool betweenLine, const std::string& uniqueStr );
    MRVIEWER_API void reorderSceneIfNeeded_();

    MRVIEWER_API void draw_object_recurse_( Object& object, const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all );

    MRVIEWER_API float drawSelectionInformation_();
    MRVIEWER_API void drawFeaturePropertiesEditor_( const std::shared_ptr<Object>& object );
    MRVIEWER_API bool drawGeneralOptions_( const std::vector<std::shared_ptr<Object>>& selectedObjs );
    MRVIEWER_API bool drawAdvancedOptions_( const std::vector<std::shared_ptr<VisualObject>>& selectedObjs, SelectedTypesMask selectedMask );

    MRVIEWER_API bool drawRemoveButton_( const std::vector<std::shared_ptr<Object>>& selectedObjs );
    MRVIEWER_API bool drawDrawOptionsCheckboxes_( const std::vector<std::shared_ptr<VisualObject>>& selectedObjs, SelectedTypesMask selectedMask );
    MRVIEWER_API bool drawDrawOptionsColors_( const std::vector<std::shared_ptr<VisualObject>>& selectedObjs );

    MRVIEWER_API virtual void draw_custom_selection_properties( const std::vector<std::shared_ptr<Object>>& selected );

    MRVIEWER_API float drawTransform_();

    std::vector<Object*> getPreSelection_( Object* meshclicked,
                                           bool isShift, bool isCtrl,
                                           const std::vector<std::shared_ptr<Object>>& selected,
                                           const std::vector<std::shared_ptr<Object>>& all );

    MRVIEWER_API virtual void drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& /*selected*/ )
    {}

    MRVIEWER_API virtual bool drawTransformContextMenu_( const std::shared_ptr<Object>& /*selected*/ ) { return false; }

    void draw_history_block_();

    void draw_open_recent_button_();

    // A virtual function for drawing of the dialog with shortcuts. It can be overriden in the inherited classes
    MRVIEWER_API virtual void drawShortcutsWindow_();
    // returns width of items in Scene Info window
    MRVIEWER_API float getSceneInfoItemWidth_( int itemCount  = 1 );
    // getting the mask of the list of selected objects
    MRVIEWER_API SelectedTypesMask calcSelectedTypesMask( const std::vector<std::shared_ptr<Object>>& selectedObjs );

    class UiRenderManagerImpl : public UiRenderManager
    {
    public:
        MRVIEWER_API void preRenderViewport( ViewportId viewport ) override;
        MRVIEWER_API void postRenderViewport( ViewportId viewport ) override;
        MRVIEWER_API BasicUiRenderTask::BackwardPassParams beginBackwardPass() override;
        MRVIEWER_API void finishBackwardPass( const BasicUiRenderTask::BackwardPassParams& params ) override;
        MRVIEWER_API bool ownsMouseHover() const override;

    private:
        // `finishBackwardPass()` sets this to true when some object's UI is hovered.
        // We then read it in `onMouseDown_()` and return true to eat that event.
        // We intentionally don't block other events with this, especially not scrolling events, as it's
        //   very annoying when the zoom breaks because you randomly hovered something.
        // It also would be unwise to block `onMouseUp_()`, as you could make some plugin stuck assuming that the mouse is held.
        bool mouseIsBlocked_ = false;
    };
    // This class helps the viewer to `renderUi()` from `IRenderObject`s.
    std::unique_ptr<UiRenderManager> uiRenderManager_;
};


// Check if menu is available and if it is, shows modal window
MRVIEWER_API void showModal( const std::string& error, NotificationType type );
inline  void showError( const std::string& error )
{
    showModal( error, NotificationType::Error );
}

} // end namespace
