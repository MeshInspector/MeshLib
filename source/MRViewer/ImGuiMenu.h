#pragma  once
#include "MRViewerPlugin.h"
#include "MRViewerEventsListener.h"
#include "MRNotificationType.h"
#include "MRSignalCombiners.h"
#include "MRShowModal.h"
#include "imgui.h"
#include "MRMesh/MRIRenderObject.h" //only for BasicUiRenderTask::BackwardPassParams
#include "MRMesh/MRFlagOperators.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRColor.h"
#include <optional>
#include <unordered_map>

// Forward declarations
struct ImGuiContext;
struct ImGuiWindow;

namespace MR
{

class ShortcutManager;
class MeshModifier;
struct UiRenderManager;
class SceneObjectsListDrawer;
class ObjectComparableWithReference;
class CombinedHistoryAction;

enum class SelectedTypesMask
{
    ObjectBit = 1 << 0,
    ObjectPointsHolderBit = 1 << 1,
    ObjectLinesHolderBit = 1 << 2,
    ObjectMeshHolderBit = 1 << 3,
    ObjectLabelBit = 1 << 4,
    ObjectMeshBit = 1 << 5,
    ObjectFeatureBit = 1 << 6,
    ObjectMeasurementBit = 1 << 7,
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
    PostResizeListener, PostRescaleListener, PostFocusListener>
{
    using ImGuiMenuMultiListener = MultiListener<
        MouseDownListener, MouseMoveListener, MouseUpListener, MouseScrollListener,
        CharPressedListener, KeyDownListener, KeyUpListener, KeyRepeatListener,
        SpaceMouseMoveListener, SpaceMouseDownListener,
        TouchpadRotateGestureBeginListener, TouchpadRotateGestureUpdateListener, TouchpadRotateGestureEndListener,
        TouchpadSwipeGestureBeginListener, TouchpadSwipeGestureUpdateListener, TouchpadSwipeGestureEndListener,
        TouchpadZoomGestureBeginListener, TouchpadZoomGestureUpdateListener, TouchpadZoomGestureEndListener,
        PostResizeListener, PostRescaleListener, PostFocusListener>;
protected:
  // Hidpi scaling to be used for text rendering.
  float hidpiScale_;

  // The ratio of the framebuffer size to the window size.
  // May be different from the hipdi scaling!
  float pixelRatio_;

  // user defined additional scaling modifier
  float userScaling_ = 1.0f;

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

  MRVIEWER_API virtual void setupShortcuts_();

  bool savedDialogPositionEnabled_{ false };

  std::weak_ptr<Object> lastRenameObj_;
  Box3f selectionLocalBox_; // updated in drawSelectionInformation_
  Box3f selectionWorldBox_;
  enum class CoordType : int
  {
      Local,
      World,
  } coordType_{ CoordType::Local };

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

  std::optional<std::pair<std::string, Vector4f>> storedColor_;
  Vector4f getStoredColor_( const std::string& str, const Color& defaultColor ) const;

  std::string searchPluginsString_;

  std::vector<std::shared_ptr<MR::MeshModifier>> modifiers_;

  enum ViewportConfigurations
  {
      Single,
      Horizontal, // left viewport, right viewport
      Vertical, // lower viewport, upper viewport
      Quad // left lower vp, left upper vp, right lower vp, right upper vp
  } viewportConfig_{ Single };

  // Tracks whether the transform block is currently being drawn in the scene info panel.
  // On the false→true transition (i.e. the block just appeared) the scene-list scroll is
  // fixed up so the selected row stays in view as the properties panel grows.
  bool transformBlockShown_{ false };
  // menu will change objects' colors in this viewport
  ViewportId selectedViewport_ = {};

  // When editing feature properties, this is the target object.
  std::weak_ptr<Object> editedFeatureObject_;
  // When editing feature properties, this is the original xf of the target object, for history purposes.
  AffineXf3f editedFeatureObjectOldXf_;

    // state for the Edit Tag modal dialog
    struct TagEditorState
    {
        std::string initName;
        std::string name;
        bool initHasFrontColor = false;
        bool hasFrontColor = false;
        ImVec4 selectedColor;
        ImVec4 unselectedColor;
    };
    TagEditorState tagEditorState_;
    // whether to open the Edit Tag modal dialog
    bool showEditTag_ = false;
    // buffer string for the tag name input widget
    std::string tagNewName_;

public:
  MRVIEWER_API static const std::shared_ptr<ImGuiMenu>& instance();

  MRVIEWER_API virtual void init(MR::Viewer *_viewer) override;

  // inits glfw and glsl backend
  MRVIEWER_API virtual void initBackend();

  // call this to validate imgui context in the begining of the frame
  MRVIEWER_API virtual void startFrame();
  // call this to draw valid imgui context at the end of the frame
  MRVIEWER_API virtual void finishFrame();

  MRVIEWER_API virtual void loadFonts( int fontSize = 13 );
  [[deprecated]] virtual void load_font( int fontSize = 13 ) { loadFonts( fontSize ); }

  MRVIEWER_API virtual void reloadFonts( int fontSize = 13 );
  [[deprecated]] virtual void reload_font( int fontSize = 13 ) { reloadFonts( fontSize ); }

  MRVIEWER_API virtual void shutdown() override;

  // Draw menu
  MRVIEWER_API virtual void draw_menu();

  MRVIEWER_API void draw_helpers();

  /// override this instead using callback_draw_viewer_window
  MRVIEWER_API virtual void drawViewerWindow();
  [[deprecated]] virtual void draw_viewer_window() { drawViewerWindow(); }

  /// override this instead using callback_draw_viewer_menu
  virtual void drawViewerWindowContent() {}

  /// override this instead using callback_draw_custom_window
  virtual void drawAdditionalWindows() {}
  [[deprecated]] virtual void draw_custom_window() { drawAdditionalWindows(); }

  [[deprecated]] MRVIEWER_API void draw_text(
      const Viewport& viewport,
      const Vector3f& pos,
      const Vector3f& normal,
      const std::string& text,
      const Color& color,
      bool clipByViewport );

  void drawLabelsWindow();
  [[deprecated]] void draw_labels_window() { drawLabelsWindow(); }

  // Computes pixel ratio for hidpi devices
  MRVIEWER_API float pixelRatio();
  [[deprecated]] float pixel_ratio() { return pixelRatio(); }

  // Computes scaling factor for hidpi devices
  MRVIEWER_API float hidpiScaling();
  [[deprecated]] float hidpi_scaling() { return hidpiScaling(); }

  MRVIEWER_API void updateScaling();

  MRVIEWER_API float menuScaling() const;
  [[deprecated]] MRVIEWER_API float menu_scaling() const;

  // returns UI scaling modifier specified by user
  float getUserScaling() const { return userScaling_; }
  // sets UI scaling modifier specified by user
  MRVIEWER_API void setUserScaling( float scaling );

  MRVIEWER_API ImGuiContext* getCurrentContext() const;

  ImGuiWindow* getLastFocusedPlugin() const { return prevFrameFocusPlugin_; };

  // opens Error / Warning / Info modal window with message text
  MRVIEWER_API virtual void showModalMessage( const std::string& msg, NotificationType msgType );

  MRVIEWER_API virtual std::filesystem::path getMenuFontPath() const;

  // setup maximum good time for frame rendering (if rendering is slower it will become red in statistics window)
  MRVIEWER_API void setDrawTimeMillisecThreshold( long long maxGoodTimeMillisec );

  // Draw scene list window with content
  MRVIEWER_API void draw_scene_list();
  // Draw scene list content only
  MRVIEWER_API void draw_scene_list_content( const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all );

  // override this to have custom "Selection Properties" window
  // draw window with content
  MRVIEWER_API virtual void draw_selection_properties( const std::vector<std::shared_ptr<Object>>& selected );
  // override this to have custom "Selection Properties" content
  // draw content only
  MRVIEWER_API virtual void draw_selection_properties_content( const std::vector<std::shared_ptr<Object>>& selected );
  // override this to have custom UI in "Selection Properties" window (under "Draw Options")

  // override this to customize appearance of collapsing headers
  MRVIEWER_API virtual bool drawCollapsingHeader_( const char* label, ImGuiTreeNodeFlags flags = 0);
  // override this to customize appearance of collapsing headers for transform block
  MRVIEWER_API virtual bool drawCollapsingHeaderTransform_();

  bool make_visualize_checkbox( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label, AnyVisualizeMaskEnum type, MR::ViewportMask viewportid, bool invert = false );
  template<typename ObjectT>
  void make_color_selector( std::vector<std::shared_ptr<ObjectT>> selectedVisualObjs, const char* label,
                            std::function<Vector4f( const ObjectT* )> getter,
                            std::function<void( ObjectT*, const Vector4f& )> setter );
  template<typename ObjType,typename ValueT>
  void make_width( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label,
                   std::function<ValueT( const ObjType* )> getter,
                   std::function<void( ObjType*, const ValueT& )> setter );

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

  std::shared_ptr<ShortcutManager> getShortcutManager() { return shortcutManager_; };

  MRVIEWER_API void add_modifier( std::shared_ptr<MR::MeshModifier> modifier );

  MRVIEWER_API void allowSceneReorder( bool allow );
  bool checkPossibilityObjectRemoval() { return allowRemoval_; };

  MRVIEWER_API void allowObjectsRemoval( bool allow );

  MRVIEWER_API void tryRenameSelectedObject();

  MRVIEWER_API void setObjectTreeState( const Object* obj, bool open );

  /// expands all `obj`s parents in tree and scroll scene tree window so selection becomes visible
  MRVIEWER_API void expandObjectTreeAndScroll( const Object* obj );

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

  MRVIEWER_API const std::shared_ptr<SceneObjectsListDrawer>& getSceneObjectsList() { return sceneObjectsList_; };

  enum class NameTagSelectionMode
  {
      // Click without modifiers, selects one object and unselects all others.
      selectOne,
      // Ctrl+Click, toggles the selection of one object.
      toggle,
  };
  using NameTagClickSignal = boost::signals2::signal<bool( Object& object, NameTagSelectionMode mode ), StopOnTrueCombiner>;
  // This is triggered whenever a name tag of an object is clicked.
  NameTagClickSignal nameTagClickSignal;
  // Behaves as if the user clicked the object name tag, by invoking `nameTagClickSignal`.
  MRVIEWER_API bool simulateNameTagClick( Object& object, NameTagSelectionMode mode );
  // This version uses the currently held keyboard modifiers instead of a custom `mode`.
  MRVIEWER_API bool simulateNameTagClickWithKeyboardModifiers( Object& object );

  using DrawSceneUiSignal = boost::signals2::signal<void( ViewportId viewportId, UiRenderParams::UiTaskList& tasks )>;
  // This is called every frame for every viewport. Use this to draw UI bits on top of the scene.
  DrawSceneUiSignal drawSceneUiSignal;

  // Scene pick should be disabled because an ImGui window is in the way.
  MRVIEWER_API bool anyImGuiWindowIsHovered() const;
  // Scene pick should be disabled because a `renderUi()` UI of some object is in the way.
  MRVIEWER_API bool anyUiObjectIsHovered() const;

    // ======== selected objects options drawing
    // getting the mask of the list of selected objects
    MRVIEWER_API SelectedTypesMask calcSelectedTypesMask( const std::vector<std::shared_ptr<Object>>& selectedObjs );
    MRVIEWER_API bool drawGeneralOptions( const std::vector<std::shared_ptr<Object>>& selectedObjs );
    MRVIEWER_API bool drawAdvancedOptions( const std::vector<std::shared_ptr<VisualObject>>& selectedObjs, SelectedTypesMask selectedMask );
    MRVIEWER_API bool drawRemoveButton( const std::vector<std::shared_ptr<Object>>& selectedObjs );
    MRVIEWER_API bool drawDrawOptionsCheckboxes( const std::vector<std::shared_ptr<VisualObject>>& selectedObjs, SelectedTypesMask selectedMask );
    MRVIEWER_API bool drawDrawOptionsColors( const std::vector<std::shared_ptr<VisualObject>>& selectedObjs );

    /// style constants used for the information panel
    struct SelectionInformationStyle
    {
        /// value text color
        Color textColor;
        /// property label color
        Color labelColor;
        /// selected value text color
        Color selectedTextColor;
        /// value item width
        float itemWidth {};
        /// value item width for two-segment field
        float item2Width {};
        /// value item width for three-segment field
        float item3Width {};
    };

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
    // Other events
    MRVIEWER_API virtual void postFocus_( bool focused ) override;

    // This function reset ImGui style to current theme and scale it by menuScaling
    // called in ImGuiMenu::postRescale_()
    MRVIEWER_API virtual void rescaleStyle_();

    MRVIEWER_API float drawSelectionInformation_();
    MRVIEWER_API void drawFeaturePropertiesEditor_( const std::shared_ptr<Object>& object );

    MRVIEWER_API void drawComparablePropertiesEditor_( ObjectComparableWithReference& object );

    /// draw additional selection information (e.g. for custom objects)
    MRVIEWER_API virtual void drawCustomSelectionInformation_( const std::vector<std::shared_ptr<Object>>& selected, const SelectionInformationStyle& style );

    MRVIEWER_API virtual void draw_custom_selection_properties( const std::vector<std::shared_ptr<Object>>& selected );

    MRVIEWER_API void drawTagInformation_( const std::vector<std::shared_ptr<Object>>& selected );

    MRVIEWER_API float drawTransform_();

    // Draws a read-only row of long-dashes ("—") matching the column layout of `UI::drag`;
    // used in drawTransform_ when the selection's transforms are not all equal.
    // `trailingLabel`, if non-null, is shown to the right of the last column (matches the
    // label rendering of the editable `UI::drag` widgets).
    MRVIEWER_API void drawMixedTransformField_( const char* labelId, int columns, const char* trailingLabel = nullptr );

    // Builds one ChangeXfAction per object and wraps them in a single CombinedHistoryAction,
    // so a multi-object transform edit goes into the history as a single undoable step.
    MRVIEWER_API std::shared_ptr<CombinedHistoryAction> makeObjectsXfHistoryAction_(
        const std::string& name,
        const std::vector<std::shared_ptr<Object>>& objs ) const;

    // Applies the same transform to every object in objs.
    MRVIEWER_API void applyXfToObjects_(
        const AffineXf3f& xf,
        const std::vector<std::shared_ptr<Object>>& objs );

    // Context menu for the transform panel. Invoked for the whole selection — caller guarantees
    // all passed objects share the same xf, so Copy/Save read from the first and Paste/Load/Reset
    // apply to every object under one combined history entry.
    MRVIEWER_API virtual bool drawTransformContextMenu_( const std::vector<std::shared_ptr<Object>>& /*selected*/ ) { return false; }

    // A virtual function for drawing of the dialog with shortcuts. It can be overriden in the inherited classes
    MRVIEWER_API virtual void drawShortcutsWindow_();
    // returns width of items in Scene Info window
    MRVIEWER_API float getSceneInfoItemWidth_( int itemCount  = 1 );

    class UiRenderManagerImpl : public UiRenderManager
    {
    public:
        MRVIEWER_API void preRenderViewport( ViewportId viewport ) override;
        MRVIEWER_API void postRenderViewport( ViewportId viewport ) override;
        MRVIEWER_API BasicUiRenderTask::BackwardPassParams beginBackwardPass( ViewportId viewport, UiRenderParams::UiTaskList& tasks ) override;
        MRVIEWER_API void finishBackwardPass( ViewportId viewport, const BasicUiRenderTask::BackwardPassParams& params ) override;

        // Which things are blocked by our `renderUi()` calls.
        BasicUiRenderTask::InteractionMask consumedInteractions{};

        // If this returns false, the event should be allowed to pass through to other plugins, even if ImGui wants to consume it.
        // Pass at most one bit at a time.
        MRVIEWER_API bool canConsumeEvent( BasicUiRenderTask::InteractionMask event ) const;
    };
    // This class helps the viewer to `renderUi()` from `IRenderObject`s.
    std::unique_ptr<UiRenderManagerImpl> uiRenderManager_;
    std::shared_ptr<SceneObjectsListDrawer> sceneObjectsList_;
};


// call if you want ImGui to take event if this key is pressed (to prevent scene reaction on key press)
MRVIEWER_API void reserveKeyEvent( ImGuiKey key );


} // end namespace
