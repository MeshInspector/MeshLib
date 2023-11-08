#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRViewer.h"
#include "MRMesh/MRChangeMeshAction.h"

namespace MR
{

/// @brief widget for surface modifying
/// @detail available 3 modes:
/// add (move surface region in direction of normal)
/// remove (move surface region in opposite direction to normal)
/// relax (relax surface region)
class MRVIEWER_CLASS SurfaceManipulationWidget
{
public:
    struct Settings
    {
        float radius = 1.f; // [1 - ...]
        float force = 10.f; // [1 - 100]
        float intensity = 50.f; // [1 - 100]
    };

    MRVIEWER_API void init( const std::shared_ptr<ObjectMesh>& objectMesh );
    MRVIEWER_API void reset();

    MRVIEWER_API void setSettings( const Settings& settings );
    MRVIEWER_API const Settings& getSettings() { return settings_; };

    MRVIEWER_API bool onMouseDown( Viewer::MouseButton button, int modifier );
    MRVIEWER_API bool onMouseUp( Viewer::MouseButton button, int modifier );
    MRVIEWER_API bool onMouseMove( int mouse_x, int mouse_y );

    MRVIEWER_API bool onKeyDown( int key, int modifier );
    MRVIEWER_API bool onKeyUp( int key, int modifier );

    MRVIEWER_API void postDraw();
private:

    void changeSurface_();
    void updateUV_( bool set );
    void updateRegion_( const Vector2f& mousePos );

    Settings settings_;

    std::shared_ptr<ObjectMesh> obj_;
    float diagonal_ = 1.f;
    Vector2f mousePos_;
    bool mouseMoved_ = false;
    VertBitSet region_;
    VertBitSet regionExpanded_; // need for proper visualization
    VertBitSet regionOld_;
    VertScalars changedValues_;
    VertScalars distances_;
    VertUVCoords uvs_;
    std::shared_ptr<ChangeMeshAction> changeMeshAction_;

    bool mousePressed_ = false;
    enum class WorkMode
    {
        Add,
        Remove,
        Relax
    } workMode_ = WorkMode::Add;
};

}
