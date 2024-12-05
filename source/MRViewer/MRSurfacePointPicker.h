#pragma once

#include "MRViewerEventsListener.h"

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRMeshTriPoint.h"
#include <MRMesh/MRColor.h>
#include "MRMesh/MRPointOnObject.h"

#include <functional>

namespace MR
{

// Widget for controlling point on surface with mouse
class MRVIEWER_CLASS SurfacePointWidget : public MultiListener<PreDrawListener, MouseDownListener, MouseMoveListener, MouseUpListener>
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
        enum class PointSizeType {
            Metrical, // point size in mm
            Pixel   // point size in pixels (will automatically be multipled by the UI scale)
        };
        // type of point positioning, look at PositionType comments for more info
        PositionType positionType{ PositionType::Faces };
        // basic color of control sphere
        Color baseColor{ Color::gray() };
        // color of control sphere when it is hovered by mouse
        Color hoveredColor{ Color::red() };
        // color of control sphere when it is in move
        Color activeColor{ { Color::red() } };
        // how to set the size of the dots in mm or in pixels.
        PointSizeType radiusSizeType{ PointSizeType::Pixel };
        // Radius of control sphere. If <= 0.0f, uses a default value (a certain fraction of the AABB diagnoal of the target object for `radiusSizeType == Metrical`, or a fixed pixel size for `radiusSizeType == Pixel`).
        // When `radiusSizeType == Pixel`, this is automatically multiplied by the current UI scale, you don't need to do that yourself.
        float radius{ 0.0f };
        // Typically, the widget does not respond to actions with a modifier.
        // If the parameter is set, then custom modifiers located in this GLFW bitmask will be ignored and the widget will work with them as usual.
        int customModifiers = 0; // GLFW modifier bitmask
        // pick_render_object parameters. Allow to use object in which pick exactly fell, instead of closer object in pick radius.
        bool pickInBackFaceObject = true;
    };

    // creates control sphere in start pos
    // returns updated pos if it was moved according to PositionType
    MRVIEWER_API const PickedPoint& create( const std::shared_ptr<VisualObject>& surface, const PointOnObject& startPos );
    MRVIEWER_API const PickedPoint& create( const std::shared_ptr<VisualObject>& surface, const PickedPoint& startPos );

    // resets whole widget
    MRVIEWER_API void reset();
    // returns object of control sphere
    const std::shared_ptr<SphereObject>& getPickSphere() const
    {
        return pickSphere_;
    }
    // get current setup of this widget
    const Parameters& getParameters() const
    {
        return params_;
    }

    // set parameters for this widget
    MRVIEWER_API void setParameters( const Parameters& params );

    // set baseColor parameter for this widget
    MRVIEWER_API void setBaseColor( const Color& color );

    /// Update the widget parameters.
    /// \param visitor - the function that takes the widget parameters and modifies them. Then the parameters are applied by \ref setParameters.
    MRVIEWER_API void updateParameters( const std::function<void ( Parameters& )>& visitor );

    // if auto hover is enabled, pick_render_object() is used
    // !note: disabling it is useful if there are many widgets, not to call `pick_render_object()` for each of them separately
    bool getAutoHover()const
    {
        return autoHover_;
    }
    void setAutoHover( bool on )
    {
        autoHover_ = on;
    }
    // function for manual enable and disable hover mode
    // use it if auto hover is disabled
    MRVIEWER_API void setHovered( bool on );

    // returns stored position of this widget
    const PickedPoint& getCurrentPosition() const
    {
        return currentPos_;
    }

    /// return current position transformed to Vector3f
    MRVIEWER_API Vector3f toVector3f() const;

    /// returns stored position as MeshTriPoint, otherwise returns invalid (default) MeshTriPoint
    MeshTriPoint getCurrentPositionMeshTriPoint() const
    {
        if ( const MeshTriPoint* triPoint = std::get_if<MeshTriPoint>( &currentPos_ ) )
            return *triPoint;
        else
            return {};
    }

    /// sets new position for the widget
    MRVIEWER_API void setCurrentPosition( const PointOnObject& pos );

    /// sets new position for the widget
    MRVIEWER_API void setCurrentPosition( const PickedPoint& pos );

    /// sets new position for the widget and returns previous position in the argument
    MRVIEWER_API void swapCurrentPosition( PickedPoint& pos );

    // this callback is called when modification starts if it is set
    void setStartMoveCallback( std::function<void( SurfacePointWidget &, const PickedPoint& )> startMove )
    {
        startMove_ = startMove;
    }
    // this callback is called on modification if it is set
    void setOnMoveCallback( std::function<void( SurfacePointWidget &, const PickedPoint& )> onMove )
    {
        onMove_ = onMove;
    }
    // this callback is called when modification ends if it is set
    void setEndMoveCallback( std::function<void( SurfacePointWidget &, const PickedPoint& )> endMove )
    {
        endMove_ = endMove;
    }

    std::shared_ptr<VisualObject>& getBaseSurface()
    {
        return baseObject_;
    }

    // returns whether is the widget moving
    [[nodiscard]] bool isOnMove() const { return isOnMove_; }

    // Checks whether the current peak is a peak in the invisible (reverse) side of the mesh or cloud point.
    [[nodiscard]] static bool isPickIntoBackFace( const std::shared_ptr<MR::VisualObject>& obj, const MR::PointOnObject& pick, const Vector3f& cameraEye );

private:
    MRVIEWER_API virtual bool onMouseDown_( MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;

    void updatePositionAndRadius_();
    void updatePositionAndRadiusMesh_( MeshTriPoint mtp );
    void updatePositionAndRadiusPoints_( const VertId& v );
    void updatePositionAndRadiusLines_( const EdgePoint& ep );

    Parameters params_;

    bool autoHover_{ true };
    bool isOnMove_{ false };
    bool isHovered_{ false };
    MRVIEWER_API void preDraw_() override;

    PickedPoint currentPos_;

    std::shared_ptr<SphereObject> pickSphere_;
    std::shared_ptr<VisualObject> baseObject_;

    boost::signals2::scoped_connection onBaseObjectWorldXfChanged_;

    std::function<void( SurfacePointWidget &, const PickedPoint& )> startMove_;
    std::function<void( SurfacePointWidget &, const PickedPoint& )> onMove_;
    std::function<void( SurfacePointWidget &, const PickedPoint& )> endMove_;

    // Depending on the type of selected size, sets the point size
    void setPointRadius_();

};



}