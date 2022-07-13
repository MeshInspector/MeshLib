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
    




    MRVIEWER_API virtual bool drawTransformContextMenu_( const std::shared_ptr<Object>& /*selected*/ ) { return false; }



protected:

    std::vector<std::shared_ptr<MR::MeshModifier>> modifiers_;


private:
    void draw_open_recent_button_();

    void draw_history_block_();


    // TODO move to independent namespace (ImGui_utils)


    std::string searchPluginsString_;

    ImVec2 mainWindowPos_;
    ImVec2 mainWindowSize_;






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

    bool showNewSelectedObjects_{ true };
};

}
