#pragma once
#include "MRViewerFwd.h"
#include "MRMouse.h"
#include "MRImGui.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRSignal.h"

namespace MR
{

/// Helper class to incorporate basic object transformation feature into plugins
/// User can move objects by dragging them, rotate by dragging with Ctrl key; scaling is disabled by default
/// To use, create class instance and call its event handlers
/// For extra features, override the `pick_` method
class MRVIEWER_CLASS MoveObjectByMouseImpl
{
public:
    MoveObjectByMouseImpl() = default;
    virtual ~MoveObjectByMouseImpl() = default;

    /// Minimum drag distance in screen pixels
    /// If cursor moved less than this value, no transform is done
    /// Default value 0 (set transformation even if mouse did not move)
    int minDistance() const { return minDistance_; }
    void setMinDistance( int minDistance ) { minDistance_ = minDistance; }

    /// Drawing callback to draw lines and tooltips
    /// Should be called from `drawDialog`
    MRVIEWER_API void onDrawDialog( float menuScaling ) const;

    /// These functions should be called from corresponding mouse handlers
    /// Or mouse drag handlers, making it work together with mouseClick signal
    /// Return true if handled (picked/moved/released)
    /// It is recommended to call `viewer->select_hovered_viewport()` before `onMouseDown`
    /// The object is transformed temporarily in `onMouseMove`
    /// `onMouseUp` finalizes the transformation and writes to undo/redo history (unless cancelled)
    MRVIEWER_API bool onMouseDown( MouseButton button, int modifiers );
    MRVIEWER_API bool onMouseMove( int x, int y );
    MRVIEWER_API bool onMouseUp( MouseButton button, int modifiers );

    /// Returns true if currently moving object(s)
    /// Return false if not active, or object picked but `minDistance` has not yet reached
    MRVIEWER_API bool isMoving() const;

    /// Stop moving the object(s). Does nothing if not moving anything
    /// Calling `onMouseUp` is not necessary after this
    /// Should be called when closing plugin etc.
    MRVIEWER_API void cancel();

    /// enables or disables adding history to viewer history store by this tool
    void enableHistory( bool history ) { historyEnabled_ = history; }

    /// returns true if appending history to viewer history store is enabled in this tool
    bool isHistoryEnabled() const { return historyEnabled_; }

protected:
    /// Transformation mode
    enum class TransformMode
    {
        None,
        Translation,
        Rotation,
        UniformScale,
        NonUniformScale,
    };

    /// if this value is > 0.0f, then Rotation and Scale will be blocked in this zone around xf center
    /// (this value IS automatically modified by menuScaling)
    float deadZonePixelRadius_{ 20.0f };

    /// This function is called from `onMouseMove` to update current active objects
    /// `objects` - list of objects to be affected by transformation
    MRVIEWER_API virtual ObjAndPick pickObjects_( std::vector<std::shared_ptr<Object>>& objects, int modifiers ) const;

    /// Helper function to determine TransformMode based on modifiers
    MRVIEWER_API virtual TransformMode modeFromPickModifiers_( int modifiers ) const;

    /// this function is called from `onMouseDown` to verify if pick should proceed, if None is returned - `onMouseDown` is canceled
    MRVIEWER_API virtual TransformMode modeFromPick_( MouseButton button, int modifiers ) const;

    /// `startPoint` - a point under cursor for transform calculation, can be the picked point or else (world coordinates)
    MRVIEWER_API virtual void setStartPoint_( const ObjAndPick& pick, Vector3f& startPoint ) const;

    /// `centerPoint` - a point that will be used as center of rotation/scaling in world space
    MRVIEWER_API virtual void setCenterPoint_( const std::vector<std::shared_ptr<Object>>& objects, Vector3f& centerPoint ) const;

    /// Helper function to calculate world bounding box for several objects
    /// Note: can be invalid (feature objects give an invalid box etc.)
    MRVIEWER_API Box3f getBbox_( const std::vector<std::shared_ptr<Object>>& objects ) const;

    AffineXf3f currentXf_;      // Transform currently applied to objects
private:
    int minDistance_ = 0;

    /// Called from `onMouseDown`
    /// Returns chosen `TransformMode` to start transformation, `None` to skip; fills the output parameters:
    /// `centerPoint` - center point for rotation (world coordinates)
    /// `startPoint` - a point under cursor for transform calculation, can be the picked point or else (world coordinates)
    /// Default implementation can be used as a reference for custom implementations
    TransformMode pick_( MouseButton button, int modifiers );

    /// one can override this function to modify derived class right after `pick_` is called
    MRVIEWER_API virtual void onPick_(
        TransformMode mode, const std::vector<std::shared_ptr<Object>>& objects,
        const Vector3f& centerPoint, const Vector3f& startPoint );

    void clear_();

    void applyCurrentXf_();

    void setVisualizeVectors_( std::vector<Vector3f> worldPoints );

    std::vector<std::shared_ptr<Object>> objects_;
    std::vector<AffineXf3f> initialXfs_;

    TransformMode transformMode_ = TransformMode::None;
    Vector2i screenStartPoint_; // onMouseDown() writes here the position of mouse when mouse dragging was started
    bool xfChanged_ = false; // it becomes true when onMouseMove changes transform of objects for the first time and optionally appends history actions
    MouseButton currentButton_ = MouseButton::NoButton;

    // Data used to calculate transform
    Vector3f worldStartPoint_;  // World point corresponding to cursor, for transform calculation
    Vector3f xfCenterPoint_;
    float viewportStartPointZ_;
    Plane3f referencePlane_;
    float angle_ = 0.f;
    float shift_ = 0.f;
    float scale_ = 1.f;

    // only check on real appending history
    bool historyEnabled_{ true };

    std::vector<ImVec2> visualizeVectors_;

    // monitors external transform change of objects during mouse moving
    std::vector<boost::signals2::scoped_connection> connections_;
    bool changingXfFromMouseMove_{ false }; // true only during setXf called from onMouseMove_
};

}
