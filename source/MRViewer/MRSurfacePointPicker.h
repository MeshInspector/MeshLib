#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRViewer.h"
#include "MRMesh/MRMeshTriPoint.h"
#include <functional>


namespace MR
{

// Widget for controlling point on surface with mouse
class MRVIEWER_CLASS SurfacePointWidget : public MultiListener<MouseDownListener, MouseMoveListener, MouseUpListener>
{
public:
    MRVIEWER_API ~SurfacePointWidget();

    enum class PositionType
    {
        Faces, // point can be in any place of surface
        FaceCenters, // point can be only in face center
        Edges, // point can be only on edges
        EdgeCeneters, // point can be only in edge center
        Verts // point can be only in vertex
    };

    struct Parameters
    {
        // type of point positioning, look at PositionType comments for more info
        PositionType positionType{ PositionType::Faces };
        // basic color of control sphere
        Color baseColor{ Color::gray() };
        // color of control sphere when it is hovered by mouse
        Color hoveredColor{ Color::red() };
        // color of control sphere when it is in move
        Color activeColor{ { Color::red() } };
        // radius of control sphere, if <= 0.0f it is equal to 5e-3*box.diagonal()
        float radius{0.0f};
    };

    // creates control sphere in start pos
    // returns updated pos if it was moved according to PositionType
    MRVIEWER_API const MeshTriPoint& create( const std::shared_ptr<ObjectMesh>& surface, const MeshTriPoint& startPos );
    // resets whole widget
    MRVIEWER_API void reset();
    // returns object of control sphere
    std::shared_ptr<SphereObject> getPickSphere() const { return pickSphere_; }
    // get current setup of this widget
    const Parameters& getParameters() const { return params_; }
    // set parameters for this widget
    MRVIEWER_API void setParameters( const Parameters& params );

    // if auto hover is enabled, pick_render_object() is used
    // !note: disabling it is useful if there are many widgets, not to call `pick_render_object()` for each of them separately
    bool getAutoHover()const { return autoHover_; }
    void setAutoHover( bool on ) { autoHover_ = on; }
    // function for manual enable and disable hover mode
    // use it if auto hover is disabled
    MRVIEWER_API void setHovered( bool on );
    
    // returns stored position of this widget
    const MeshTriPoint& getCurrentPosition() const { return currentPos_; }

    // this callback is called when modification starts if it is set
    void setStartMoveCallback( std::function<void( const MeshTriPoint& )> startMove ) { startMove_ = startMove; }
    // this callback is called on modification if it is set
    void setOnMoveCallback( std::function<void( const MeshTriPoint& )> onMove ) { onMove_ = onMove; }
    // this callback is called when modification ends if it is set
    void setEndMoveCallback( std::function<void( const MeshTriPoint& )> endMove ) { endMove_ = endMove; }

private:
    MRVIEWER_API virtual bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;

    void updatePositionAndRadius_();

    Parameters params_;

    bool autoHover_{ true };
    bool isOnMove_{ false };
    bool isHovered_{ false };
    MeshTriPoint currentPos_;

    std::shared_ptr<SphereObject> pickSphere_;
    std::shared_ptr<ObjectMesh> baseSurface_;

    std::function<void( const MeshTriPoint& )> startMove_;
    std::function<void( const MeshTriPoint& )> onMove_;
    std::function<void( const MeshTriPoint& )> endMove_;
};

}