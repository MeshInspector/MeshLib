#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRViewer.h"
#include <boost/signals2/signal.hpp>
#include <functional>

namespace MR
{

// Visual widget to modify transform
// present in scene (ancillary), subscribes to viewer events
class MRVIEWER_CLASS ObjectTransformWidget : public MultiListener<MouseDownListener, MouseMoveListener, MouseUpListener, PreDrawListener, DrawListener>
{
public:
    enum Axis { X, Y, Z, Count };
    enum TransformMode
    {
        RotX = 0x1,
        RotY = 0x2,
        RotZ = 0x4,
        MoveX = 0x8,
        MoveY = 0x10,
        MoveZ = 0x20
    };
    // Creates transform widget around given box and applies given xf
    // subscribes to viewer events
    MRVIEWER_API void create( const Box3f& box, const AffineXf3f& xf );
    // Removes widget from scene and clears all widget objects
    // unsubscribes from viewer events
    MRVIEWER_API void reset();

    // get current width of widget controls
    // negative value means that controls are not setup
    float getWidth() const { return width_; }
    // get current radius of widget controls
    // negative value means that controls are not setup
    float getRadius() const { return radius_; }

    // set width for this widget
    MRVIEWER_API void setWidth( float width );
    // set radius for this widget
    MRVIEWER_API void setRadius( float radius );

    // Returns current transform mode mask
    uint8_t getTransformModeMask() const { return transformModeMask_; }
    // Sets transform mode mask (enabling or disabling corresponding widget controls)
    MRVIEWER_API void setTransformMode( uint8_t mask );

    // Enables or disables pick through mode, in this mode controls will be picked even if they are occluded by other objects
    void setPickThrough( bool on ) { pickThrough_ = on; }
    bool getPickThrough() const { return pickThrough_; }

    // Transform operation applying to object while dragging an axis. This parameter does not apply to active operation.
    enum AxisTransformMode
    {
        // object moves along an axis
        AxisTranslation,
        // object inflates or deflates along an axis depending on drag direction (away from center or toward center respectively)
        AxisScaling,
        // object inflates or deflates along all axes depending on drag direction (away from center or toward center respectively)
        UniformScaling,
    };
    // Returns current axis transform mode (translate/scale object while dragging an axis)
    AxisTransformMode getAxisTransformMode() const { return axisTransformMode_; };
    // Sets current axis transform mode (translate/scale object while dragging an axis)
    void setAxisTransformMode( AxisTransformMode mode ) { axisTransformMode_ = mode; };

    // Returns root object of widget
    std::shared_ptr<Object> getRootObject() const { return controlsRoot_; }

    // Changes controls xf (controls will affect object in basis of new xf)
    // note that rotation is applied around 0 coordinate in world space, so use xfAround to process rotation around user defined center
    MRVIEWER_API void setControlsXf( const AffineXf3f& xf );
    MRVIEWER_API AffineXf3f getControlsXf() const;

    // Returns threshold dot value (this value is duty for hiding widget controls that have small projection on screen)
    float getThresholdDot() const { return thresholdDot_; }
    // Sets threshold dot value (this value is duty for hiding widget controls that have small projection on screen)
    void setThresholdDot( float thresholdDot ) { thresholdDot_ = thresholdDot; }

    // Subscribes to object visibility, and behave like its child
    // if obj argument is null, stop following
    MRVIEWER_API void followObjVisibility( const std::weak_ptr<Object>& obj );

    // Sets callback that will be called in draw function during scaling with current scale arg
    void setScaleTooltipCallback( std::function<void( float )> callback ) { scaleTooltipCallback_ = callback; }
    // Sets callback that will be called in draw function during translation with current shift arg
    void setTranslateTooltipCallback( std::function<void( float )> callback ) { translateTooltipCallback_ = callback; }
    // Sets callback that will be called in draw function during rotation with current angle in rad
    void setRotateTooltipCallback( std::function<void( float )> callback ) { rotateTooltipCallback_ = callback; }

    // Sets callback that will be called when modification of widget stops
    void setStopModifyCallback( std::function<void()> callback ) { stopModifyCallback_ = callback; }
    // Sets callback that will be called when modification of widget starts
    void setStartModifyCallback( std::function<void()> callback ) { startModifyCallback_ = callback; }
    // Sets callback that will be called when widget gets addictive transform
    void setAddXfCallback( std::function<void( const AffineXf3f& )> callback ) { addXfCallback_ = callback; }
    // Sets callback that will be called when widget gets addictive transform
    // The callback should return true to approve transform and false to reject it
    void setApproveXfCallback( std::function<bool( const AffineXf3f& )> callback ) { approveXfCallback_ = callback; }
private:
    MRVIEWER_API virtual bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;
    MRVIEWER_API virtual void preDraw_() override;
    MRVIEWER_API virtual void draw_() override;

    void passiveMove_();
    void activeMove_( bool press = false );

    void processScaling_( Axis ax, bool press );
    void processTranslation_( Axis ax, bool press );
    void processRotation_( Axis ax, bool press );

    std::weak_ptr<Object> visibilityParent_;
    std::shared_ptr<ObjectMesh> currentObj_;

    void updateVisualTransformMode_( uint8_t showMask, ViewportMask viewportMask );

    void setActiveLineFromPoints_( const std::vector<Vector3f>& points );

    void addXf_( const AffineXf3f& addXf );
    void stopModify_();

    int findCurrentObjIndex_() const;

    void makeControls_();

    float radius_{ -1.0f };
    float width_{ -1.0f };

    // main object that holds all other controls
    std::shared_ptr<Object> controlsRoot_;
    std::array<std::shared_ptr<ObjectMesh>, size_t( Axis::Count )> translateControls_;
    std::array<std::shared_ptr<ObjectMesh>, size_t( Axis::Count )> rotateControls_;

    // if active line is visible, other lines are not
    std::shared_ptr<ObjectLines> activeLine_;
    std::array<std::shared_ptr<ObjectLines>, size_t( Axis::Count )> translateLines_;
    std::array<std::shared_ptr<ObjectLines>, size_t( Axis::Count )> rotateLines_;

    Vector3f center_;

    AxisTransformMode axisTransformMode_{ AxisTranslation };

    enum ActiveEditMode
    {
        TranslationMode,
        ScalingMode,
        UniformScalingMode,
        RotationMode,
    };
    ActiveEditMode activeEditMode_{ TranslationMode };

    // store original object's scaled transform for proper controls' uniform scaling calculation
    AffineXf3f scaledXf_;
    Matrix3f objScale_;

    Vector3f prevScaling_;
    Vector3f startTranslation_;
    Vector3f prevTranslation_;
    AffineXf3f startRotXf_;
    float startAngle_ = 0;
    float accumAngle_ = 0;

    uint8_t transformModeMask_ = 0x3f;
    float thresholdDot_{ 0.0f };
    bool picked_{ false };
    bool pickThrough_{ false };

    std::function<void( float )> scaleTooltipCallback_;
    std::function<void( float )> translateTooltipCallback_;
    std::function<void( float )> rotateTooltipCallback_;

    std::function<void()> startModifyCallback_;
    std::function<void()> stopModifyCallback_;
    std::function<void( const AffineXf3f& )> addXfCallback_;
    std::function<bool( const AffineXf3f& )> approveXfCallback_;
    bool approvedChange_ = true; // if controlsRoot_ xf changed without approve, user modification stops
    boost::signals2::connection xfValidatorConnection_;
};

}