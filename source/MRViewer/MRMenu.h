#pragma once
#include "ImGuiMenu.h"

#include "imgui.h"
#include "exports.h"


namespace MR
{


class MRVIEWER_CLASS Menu : public MR::ImGuiMenu
{
public:
    MRVIEWER_API virtual void init( MR::Viewer *_viewer ) override;

    // this fuction will be called for resizing or updating fonts
    // if you need to customize the whole fonts reloading process - override reload_font function
    
    

    

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

    



private:





    // TODO move to independent namespace (ImGui_utils)


    








    // maximum good time for frame rendering milliseconds (25 ms ~ 40 FPS)
    long long frameTimeMillisecThreshold_{ 25 };




    // struct to specify scene reorder params

    // Used not to accumulate plugins each frame (also sorts tab plugins by special string)
   

    bool showNewSelectedObjects_{ true };
};

}
