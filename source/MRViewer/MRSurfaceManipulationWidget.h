#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRViewer.h"
#include "MRMesh/MRChangeMeshAction.h"
#include <chrono>

namespace MR
{

/// @brief widget for surface modifying
/// @detail available 3 modes:
/// add (move surface region in direction of normal)
/// remove (move surface region in opposite direction to normal)
/// relax (relax surface region)
class MRVIEWER_CLASS SurfaceManipulationWidget :
    public MultiListener<MouseDownListener, MouseMoveListener, MouseUpListener>
{
public:
    /// widget work modes
    enum class WorkMode
    {
        Add,
        Remove,
        Relax
    };

    /// Mesh change settings
    struct Settings
    {
        WorkMode workMode = WorkMode::Add;
        float radius = 1.f; // radius of editing region
        float relaxForce = 0.2f; // speed of relaxing, typical values (0.0, 0.5]
        float editForce = 1.f; // the force of changing mesh
        float sharpness = 50.f; // effect of force on points far from center editing area. [1 - 100]
        bool relaxAfterEdit = true; // relax modified area after editing is complete. uses relaxForce
    };

    /// initialize widget according ObjectMesh
    MRVIEWER_API void init( const std::shared_ptr<ObjectMesh>& objectMesh );
    /// reset widget state
    MRVIEWER_API void reset();

    /// set widget settings (mesh change settings)
    MRVIEWER_API void setSettings( const Settings& settings );
    /// get widget settings 
    MRVIEWER_API const Settings& getSettings() { return settings_; };

    // mimum radius of editing area.
    MRVIEWER_API float getMinRadius() { return minRadius_; };

private:
    /// start modifying mesh surface
    MRVIEWER_API bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    /// stop modifying mesh surface, generate history action
    MRVIEWER_API bool onMouseUp_( Viewer::MouseButton button, int modifier ) override;
    /// update
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;

    void changeSurface_();
    void updateUV_( bool set );
    void updateRegion_( const Vector2f& mousePos );

    Settings settings_;

    std::shared_ptr<ObjectMesh> obj_;
    float diagonal_ = 1.f;
    float minRadius_ = 1.f;
    Vector2f mousePos_;
    bool mouseMoved_ = false;
    VertBitSet region_;
    VertBitSet regionExpanded_; // need for proper visualization
    VertBitSet regionChanged_;
    VertScalars pointsShift_;
    VertScalars distances_;
    VertUVCoords uvs_;
    std::shared_ptr<ChangeMeshAction> changeMeshAction_;
    std::shared_ptr<ObjectMesh> oldMesh_;
    bool firstInit_ = true; // need to save settings in re-initial

    bool mousePressed_ = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> timePoint_;
};

}
