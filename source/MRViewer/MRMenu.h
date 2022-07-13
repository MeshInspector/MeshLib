#pragma once
#include "ImGuiMenu.h"
#include "MRStatePlugin.h"
#include "imgui.h"
#include "exports.h"
#include <unordered_map>

namespace MR
{
class MeshModifier;

class MRVIEWER_CLASS Menu : public MR::ImGuiMenu
{
public:
    MRVIEWER_API virtual void init( MR::Viewer *_viewer ) override;

    // this fuction will be called for resizing or updating fonts
    // if you need to customize the whole fonts reloading process - override reload_font function
    MRVIEWER_API void draw_mr_menu();
    // Draw scene list window with content
    MRVIEWER_API void draw_scene_list();
    // Draw scene list content only
    MRVIEWER_API void draw_scene_list_content( const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all );

    MRVIEWER_API void draw_custom_plugins();

    // change object tree node to open/closed on next frame
    MRVIEWER_API void setObjectTreeState( const Object* obj, bool open );

    /// try to start rename selected obbject if it possible
    MRVIEWER_API void tryRenameSelectedObject();

    // allow or forbid removal of objects from scene
    MRVIEWER_API void allowObjectsRemoval( bool allow );

    /// return possibility of removing selected objects
    bool checkPossibilityObjectRemoval() { return allowRemoval_; };

    // allow or forbid scene reordering
    MRVIEWER_API void allowSceneReorder( bool allow );



    // override this to have custom "Selection Properties" window
    // draw window with content
    MRVIEWER_API virtual void draw_selection_properties( std::vector<std::shared_ptr<Object>>& selected );
    // override this to have custom "Selection Properties" content
    // draw content only
    MRVIEWER_API virtual void draw_selection_properties_content( std::vector<std::shared_ptr<Object>>& selected );
    // override this to have custom UI in "Selection Properties" window (under "Draw Options")
    MRVIEWER_API virtual void draw_custom_selection_properties( const std::vector<std::shared_ptr<Object>>& selected );
    // override this to have custom UI in "Scene" window (under opened(expanded) object line)
    MRVIEWER_API virtual void draw_custom_tree_object_properties( Object& obj );

    MRVIEWER_API void add_modifier( std::shared_ptr<MR::MeshModifier> modifier );

    // should return path of font that will be used in menu
    

    // setup maximum good time for frame rendering (if rendering is slower it will become red in statistics window)
    MRVIEWER_API void setDrawTimeMillisecThreshold( long long maxGoodTimeMillisec );

    std::shared_ptr<ShortcutManager> getShortcutManager() { return shortcutManager_; };
    
    // set show selected objects state (enable / disable)
    void setShowNewSelectedObjects( bool show ) { showNewSelectedObjects_ = show; };
    // get show selected objects state (enable / disable)
    bool getShowNewSelectedObjects() { return showNewSelectedObjects_; };

protected:
    // Keyboard IO
    MRVIEWER_API virtual bool onCharPressed_( unsigned key, int modifiers ) override;
    MRVIEWER_API virtual bool onKeyDown_( int key, int modifiers ) override;
    MRVIEWER_API virtual bool onKeyRepeat_( int key, int modifiers ) override;
    // add ranges (that will be used in menu) to builder 
    

    MRVIEWER_API float drawSelectionInformation_();
    MRVIEWER_API float drawTransform_();

    // payload object will be moved
    MRVIEWER_API void makeDragDropSource_( const std::vector<std::shared_ptr<Object>>& payload );
    // "target" and "before" are "to" and "before" of SceneReorder struct
    // betweenLine - if true requires to draw line (between two objects in tree, for ImGui to have target)
    // counter - unique number of object in tree (needed for ImGui to differ new lines)
    MRVIEWER_API void makeDragDropTarget_( Object& target, bool before, bool betweenLine, int counter );
    MRVIEWER_API void reorderSceneIfNeeded_();

    MRVIEWER_API void draw_object_recurse_( Object& object, const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all, int& counter );
    MRVIEWER_API virtual void drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& /*selected*/ ) {}

    MRVIEWER_API virtual void drawTransformContextMenu_( const std::shared_ptr<Object>& /*selected*/ ) {}

    // override this to customize prefix for objects in scene
    MRVIEWER_API virtual void drawCustomObjectPrefixInScene_( const Object& ) {}

    MRVIEWER_API bool drawRemoveButton_( const std::vector<std::shared_ptr<Object>>& selectedObjs );
    MRVIEWER_API bool drawDrawOptionsCheckboxes_( const std::vector<std::shared_ptr<VisualObject>>& selectedObjs );
    MRVIEWER_API bool drawDrawOptionsColors_( const std::vector<std::shared_ptr<VisualObject>>& selectedObjs );
    MRVIEWER_API bool drawGeneralOptions_( const std::vector<std::shared_ptr<Object>>& selectedObjs );

protected:

    std::vector<std::shared_ptr<MR::MeshModifier>> modifiers_;

    std::unordered_map<const Object*, bool> sceneOpenCommands_;

private:
    void draw_open_recent_button_();

    void draw_history_block_();

    std::vector<Object*> getPreSelection_( Object* meshclicked,
                                           bool isShift, bool isCtrl,
                                           const std::vector<std::shared_ptr<Object>>& selected, 
                                           const std::vector<std::shared_ptr<Object>>& all );

    // TODO move to independent namespace (ImGui_utils)
    bool make_checkbox( const char* label, bool& checked, bool mixed );
    bool make_visualize_checkbox( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label, unsigned type, MR::ViewportMask viewportid );
    template<typename ObjectT>
    void make_color_selector( std::vector<std::shared_ptr<ObjectT>> selectedVisualObjs, const char* label,
                              std::function<Vector4f( const ObjectT* )> getter,
                              std::function<void( ObjectT*, const Vector4f& )> setter );
    void make_width( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label,
                     std::function<float( const ObjectLinesHolder* )> getter,
                     std::function<void( ObjectLinesHolder*, const float& )> setter );


    std::string searchPluginsString_;

    ImVec2 mainWindowPos_;
    ImVec2 mainWindowSize_;
    ImVec2 sceneWindowPos_;
    ImVec2 sceneWindowSize_;

    bool xfHistUpdated_{ false };

    Box3f selectionBbox_; // updated in drawSelectionInformation_

    std::weak_ptr<Object> lastRenameObj_;

    bool uniformScale_{true};

    bool allowRemoval_{ true };
    bool allowSceneReorder_{ true };

    std::optional<std::pair<std::string, Vector4f>> storedColor_;
    Vector4f getStoredColor_( const std::string& str, const Color& defaultColor ) const;

    // maximum good time for frame rendering milliseconds (25 ms ~ 40 FPS)
    long long frameTimeMillisecThreshold_{ 25 };

    enum ViewportConfigurations
    {
        Single,
        Horizontal, // left viewport, right viewport
        Vertical, // lower viewport, upper viewport
        Quad // left lower vp, left upper vp, right lower vp, right upper vp
    } viewportConfig_{Single};


    // struct to specify scene reorder params
    struct SceneReorder
    {
        std::vector<Object*> who; // object that will be moved
        Object* to{nullptr}; // address object
        bool before{ false }; // if false "who" will be attached to "to" as last child, otherwise "who" will be attached to "to"'s parent as child before "to"
    } sceneReorderCommand_;

    // Used not to accumulate plugins each frame (also sorts tab plugins by special string)
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

    bool dragTrigger_ = false;
    bool clickTrigger_ = false;

    bool showNewSelectedObjects_{ true };
};

}
