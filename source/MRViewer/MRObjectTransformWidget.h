#pragma once

#include "MRViewerEventsListener.h"
#include "MRMesh/MRFlagOperators.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRHistoryAction.h"
#include "MRMesh/MRViewportProperty.h"
#include "MRMesh/MRAxis.h"
#include <MRMesh/MRObject.h>
#include <MRMesh/MRColor.h>
#include "MRMesh/MRSignal.h"
#include <array>
#include <functional>
#include <string>

namespace MR
{

enum class ControlBit
{
    None = 0,
    RotX = 0x1,
    RotY = 0x2,
    RotZ = 0x4,
    RotMask = RotX | RotY | RotZ,
    MoveX = 0x8,
    MoveY = 0x10,
    MoveZ = 0x20,
    MoveMask = MoveX | MoveY | MoveZ,
    FullMask = RotMask | MoveMask
};
MR_MAKE_FLAG_OPERATORS( ControlBit )

// This lambda is called in each frame, and returns transform mode mask for this frame in given viewport
// if not set, full mask is return
using TransformModesValidator = std::function<ControlBit( const Vector3f& center, const AffineXf3f& xf, ViewportId )>;


// Interface class for ObjectTransformWidget custom visualization
class MRVIEWER_CLASS ITransformControls
{
public:
    virtual ~ITransformControls() = default;

    // get center of the widget in local space
    const Vector3f& getCenter() const { return center_; }
    MRVIEWER_API void setCenter( const Vector3f& center );

    // should return current radius of the widget
    virtual float getRadius() const { return 1.0f; }

    // This lambda is called in each frame, and returns transform mode mask for this frame in given viewport
    // if not set, full mask is return
    void setTransformModesValidator( TransformModesValidator validator ) { validator_ = validator; }

    // Enables or disables pick through mode, in this mode controls will be picked even if they are occluded by other objects
    void setPickThrough( bool on ) { pickThrough_ = on; }
    bool getPickThrough() const { return pickThrough_; }

    // Returns currently hovered control
    ControlBit getHoveredControl() const { return hoveredControl_; }

    // Called once on widget created to init internal objects
    virtual void init( std::shared_ptr<Object> parent ) = 0;
    // Called right after init and can be called on some internal actions to recreate
    // objects visualization
    virtual void update() = 0;
    // Called for hover checks
    void hover() { hoveredControl_ = hover_( pickThrough_ ); }
    // This is called to stop drawing active visualization when modification is stopped
    void stopModify() { stopModify_(); hover(); }

    // Called each frame for each viewport to update available transformation modes
    MRVIEWER_API void updateVisualTransformMode( ControlBit showMask, ViewportMask viewportMask, const AffineXf3f& xf );

    // One have to implement these functions to have visualization of translation and rotation
    virtual void updateTranslation( Axis ax, const Vector3f& startMove, const Vector3f& endMove ) = 0;
    // xf - widget current xf
    virtual void updateRotation( Axis ax, const AffineXf3f& xf, float startAngle, float endAngle ) = 0;

    // build-in history action class for change center
    class ChangeCenterAction : public HistoryAction
    {
    public:
        ChangeCenterAction( const std::string& name, ITransformControls& controls ) :
            controls_{ controls },
            name_{ name }{ center_ = controls.getCenter(); }

        virtual std::string name() const override { return name_; }

        virtual void action( HistoryAction::Type ) override
        {
            auto center = controls_.getCenter();
            controls_.setCenter( center_ );
            center_ = center;
        }

        [[nodiscard]] virtual size_t heapBytes() const override { return name_.capacity(); }

    private:
        ITransformControls& controls_;
        Vector3f center_;
        std::string name_;
    };
protected:
    // one have to implement this function
    // it can change internal visualization and return currently hovered control
    virtual ControlBit hover_( bool pickThrough ) = 0;
    // one have to implement this function
    // it can change internal visualization
    // called when modification is stopped
    virtual void stopModify_() = 0;
    // one have to implement this function
    // it can be called in each frame (each viewport if needed) to update transform mode in different viewports
    virtual void updateVisualTransformMode_( ControlBit showMask, ViewportMask viewportMask ) = 0;
private:
    Vector3f center_;

    ControlBit hoveredControl_{ ControlBit::None };
    bool pickThrough_{ false };
    TransformModesValidator validator_;
};

// Basic implementation of ITransformControls
class MRVIEWER_CLASS TransformControls : public ITransformControls
{
public:
    struct MRVIEWER_CLASS VisualParams
    {
        // updates radius and width with given box
        MRVIEWER_API void update( const Box3f& box );
        // negative radius value means that controls are not setup
        float radius{ -1.0f };
        // negative width value means that controls are not setup
        float width{ -1.0f };
        /// the product of this factor and width gives cone radius of the arrows
        float coneRadiusFactor{ 1.35f };
        /// the product of this factor and width gives cone size of the arrows
        float coneSizeFactor{ 2.2f };
        /// extension of the translation line in the negative direction relative to the radius
        float negativeLineExtension{ 1.15f };
        /// extension of the translation line in the positive direction relative to the radius
        float positiveLineExtension{ 1.3f };
        /// colors of widget
        std::array<Color, size_t( Axis::Count )> rotationColors{ Color::red(),Color::green(),Color::blue() };
        std::array<Color, size_t( Axis::Count )> translationColors{ Color::red(),Color::green(),Color::blue() };
        Color helperLineColor{ Color::black() };
        Color activeLineColor{ Color::white() };
    };
    MRVIEWER_API void setVisualParams( const VisualParams& params );
    const VisualParams& getVisualParams() const { return params_; }

    MRVIEWER_API virtual ~TransformControls();

    MRVIEWER_API virtual void init( std::shared_ptr<Object> parent ) override;
    MRVIEWER_API virtual void update() override;

    virtual float getRadius() const override { return params_.radius; }
    // get current radius of widget controls
    // negative value means that controls are not setup
    MRVIEWER_API void setRadius( float radius );
    // get current width of widget controls
    // negative value means that controls are not setup
    float getWidth() const { return params_.width; }
    // set width for this widget
    MRVIEWER_API void setWidth( float width );

    MRVIEWER_API virtual void updateTranslation( Axis ax, const Vector3f& startMove, const Vector3f& endMove ) override;
    MRVIEWER_API virtual void updateRotation( Axis ax, const AffineXf3f& xf, float startAngle, float endAngle ) override;

    // returns TransformModesValidator by threshold dot value (this value is duty for hiding widget controls that have small projection on screen)
    MRVIEWER_API static TransformModesValidator ThresholdDotValidator( float thresholdDot );
private:
    MRVIEWER_API virtual ControlBit hover_( bool pickThrough ) override;
    MRVIEWER_API virtual void stopModify_() override;
    MRVIEWER_API virtual void updateVisualTransformMode_( ControlBit showMask, ViewportMask viewportMask ) override;

    VisualParams params_;

    // Control objects
    std::array<std::shared_ptr<ObjectMesh>, size_t( Axis::Count )> translateControls_;
    std::array<std::shared_ptr<ObjectMesh>, size_t( Axis::Count )> rotateControls_;

    // if active line is visible, other lines are not
    std::shared_ptr<ObjectLines> activeLine_;
    std::array<std::shared_ptr<ObjectLines>, size_t( Axis::Count )> translateLines_;
    std::array<std::shared_ptr<ObjectLines>, size_t( Axis::Count )> rotateLines_;

    std::shared_ptr<ObjectMesh> hoveredObject_;
    int findHoveredIndex_() const;
    void setActiveLineFromPoints_( const Contour3f& points );
};

// Visual widget to modify transform
// present in scene (ancillary), subscribes to viewer events
class MRVIEWER_CLASS ObjectTransformWidget : public MultiListener<MouseDownListener, MouseMoveListener, MouseUpListener, PreDrawListener, PostDrawListener>
{
public:
    // Creates transform widget around given box and applies given xf
    // subscribes to viewer events
    // controls: class that is responsible for visualization
    // if controls is empty default TransformControls is used
    MRVIEWER_API void create( const Box3f& box, const AffineXf3f& xf, std::shared_ptr<ITransformControls> controls = {} );
    // Removes widget from scene and clears all widget objects
    // unsubscribes from viewer events
    MRVIEWER_API void reset();

    // Returns current transform mode mask
    ControlBit getTransformModeMask( ViewportId id = {} ) const { return transformModeMask_.get( id ); }
    // Sets transform mode mask (enabling or disabling corresponding widget controls)
    MRVIEWER_API void setTransformMode( ControlBit mask, ViewportId id = {} );

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

    // Returns controls object, that visualize widget
    std::shared_ptr<ITransformControls> getControls() const { return controls_; }
    template<typename T>
    std::shared_ptr<T> getControlsAs() const { return std::dynamic_pointer_cast< T >( controls_ ); }

    // Changes controls xf (controls will affect object in basis of new xf)
    // note that rotation is applied around 0 coordinate in world space, so use xfAround to process rotation around user defined center
    // non-uniform scale will be converted to uniform one based on initial box diagonal
    MRVIEWER_API void setControlsXf( const AffineXf3f& xf, ViewportId id = {} );
    MRVIEWER_API AffineXf3f getControlsXf( ViewportId id = {} ) const;

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
    // Sets callback that will be called when widget gets additive transform
    void setAddXfCallback( std::function<void( const AffineXf3f& )> callback ) { addXfCallback_ = callback; }
    // Sets callback that will be called when widget gets additive transform
    // The callback should return true to approve transform and false to reject it
    void setApproveXfCallback( std::function<bool( const AffineXf3f& )> callback ) { approveXfCallback_ = callback; }

    // History action for TransformWidget
    class ChangeXfAction : public HistoryAction
    {
    public:
        ChangeXfAction( const std::string& name, ObjectTransformWidget& widget ) :
            widget_{ widget },
            name_{ name }
        {
            if ( widget_.controlsRoot_ )
            {
                xf_ = widget_.controlsRoot_->xfsForAllViewports();
                scaledXf_ = widget_.scaledXf_;
            }
        }

        virtual std::string name() const override
        {
            return name_;
        }

        virtual void action( HistoryAction::Type ) override
        {
            if ( !widget_.controlsRoot_ )
                return;
            auto tmpXf = widget_.controlsRoot_->xfsForAllViewports();
            widget_.controlsRoot_->setXfsForAllViewports( xf_ );
            xf_ = tmpXf;

            std::swap( scaledXf_, widget_.scaledXf_ );
        }

        [[nodiscard]] virtual size_t heapBytes() const override
        {
            return name_.capacity();
        }

    private:
        ObjectTransformWidget& widget_;
        ViewportProperty<AffineXf3f> xf_;
        ViewportProperty<AffineXf3f> scaledXf_;
        std::string name_;
    };
private:
    MRVIEWER_API virtual bool onMouseDown_( MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;
    MRVIEWER_API virtual void preDraw_() override;
    MRVIEWER_API virtual void postDraw_() override;

    void activeMove_( bool press = false );

    void processScaling_( Axis ax, bool press );
    void processTranslation_( Axis ax, bool press );
    void processRotation_( Axis ax, bool press );

    void setControlsXf_( const AffineXf3f& xf, bool updateScaled, ViewportId id = {} );

    std::weak_ptr<Object> visibilityParent_;

    // undiformAddXf - for ActiveEditMode::ScalingMode only, to scale widget uniformly
    void addXf_( const AffineXf3f& addXf );
    void stopModify_();

    // main object that holds all other controls
    std::shared_ptr<Object> controlsRoot_;
    std::shared_ptr<ITransformControls> controls_;

    AxisTransformMode axisTransformMode_{ AxisTranslation };

    enum ActiveEditMode
    {
        TranslationMode,
        ScalingMode,
        UniformScalingMode,
        RotationMode,
    };
    ActiveEditMode activeEditMode_{ TranslationMode };

    // Initial box diagonal vector (before transformation),
    // it is needed to correctly convert non-uniform scaling to uniform one and apply it to this widget
    Vector3f boxDiagonal_;
    // same as controlsRoot_->xf() but with non uniform scaling applied
    ViewportProperty<AffineXf3f> scaledXf_;
    // this is needed for tooltip only
    float currentScaling_ = 1.0f;

    Vector3f prevScaling_;
    Vector3f startTranslation_;
    Vector3f prevTranslation_;
    float accumShift_ = 0;

    float startAngle_ = 0;
    float accumAngle_ = 0;

    ViewportProperty<ControlBit> transformModeMask_{ ControlBit::FullMask };
    bool picked_{ false };

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
