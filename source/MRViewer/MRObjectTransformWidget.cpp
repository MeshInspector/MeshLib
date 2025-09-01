#include "MRObjectTransformWidget.h"
#include "MRMouseController.h"
#include "MRViewport.h"
#include "MRViewer.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRArrow.h"
#include "MRMesh/MRTorus.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRIntersection.h"
#include "MRMesh/MR2to3.h"
#include "MRMesh/MRMatrix3Decompose.h"
#include "MRMesh/MRPolyline.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"

namespace
{
constexpr std::array<MR::Vector3f, 3> baseAxis =
{ MR::Vector3f::plusX(),MR::Vector3f::plusY(),MR::Vector3f::plusZ() };

using namespace MR;

float findAngleDegOfPick( const Vector3f& center, const Vector3f& zeroPoint, const Vector3f& norm,
                          const Line3f& ray, Viewport& vp, const Vector3f& vpPoint )
{
    Plane3f plane1 = Plane3f::fromDirAndPt( norm, center ).normalized();
    Plane3f plane2 = Plane3f::fromDirAndPt( cross( ray.d, center - ray.p ), center ).normalized();
    auto centerVp = vp.projectToViewportSpace( center );
    auto centerRay = vp.unprojectPixelRay( to2dim( centerVp ) );
    bool parallel = std::abs( dot( centerRay.d.normalized(), norm.normalized() ) ) < 0.25f;
    auto planeIntersectionLine = intersection( plane1, plane2 );
    auto radiusVec = ( zeroPoint - center );
    if ( parallel || !planeIntersectionLine )
    {
        auto pickProjS = plane1.project( ray.p );
        auto pickProjF = plane1.project( ray.p + ray.d );

        auto centerProjOnPickLine = Line3f( pickProjS, pickProjF - pickProjS ).project( center );

        auto radiusSq = radiusVec.lengthSq();
        auto line = center - centerProjOnPickLine;
        auto lineSq = line.lengthSq();
        if ( lineSq >= radiusSq )
            planeIntersectionLine = Line3f( centerProjOnPickLine, line );
        else
        {
            auto pInter = centerProjOnPickLine + ( pickProjS - pickProjF ).normalized() *
                std::sqrt( radiusSq - lineSq );
            planeIntersectionLine = Line3f( pInter, center - pInter );
        }
        parallel = true;
    }
    auto vec2 = radiusVec.normalized();
    auto angleSign = mixed( norm.normalized(), planeIntersectionLine->d, vec2 ) < 0.0f ? 1.0f : -1.0f;
    auto angleRes = angleSign * angle( planeIntersectionLine->d.normalized(), vec2 );
    auto sinA = std::sin( angleRes );
    auto cosA = std::cos( angleRes );
    auto crossVec = cross( norm.normalized(), vec2 ).normalized();
    auto p1 = center + cosA * vec2 + sinA * crossVec;
    auto p2 = center - cosA * vec2 - sinA * crossVec;

    auto p1Vp = vp.projectToViewportSpace( p1 );
    auto p2Vp = vp.projectToViewportSpace( p2 );
    auto dist1 = p1Vp - vpPoint;
    auto dist2 = p2Vp - vpPoint;
    float diff = 0.0f;
    if ( !parallel )
        diff = to2dim( dist1 ).lengthSq() - to2dim( dist2 ).lengthSq();
    else
        diff = dist1.z - dist2.z;

    if ( diff > 0.0f || parallel )
    {
        if ( angleRes < 0.0f )
            return angleRes + PI_F;
        return angleRes - PI_F;
    }
    return angleRes;
}
}

namespace MR
{

void ObjectTransformWidget::create( const Box3f& box, const AffineXf3f& worldXf, std::shared_ptr<ITransformControls> controls )
{
    if ( controlsRoot_ )
        reset();

    boxDiagonal_ = box.size();
    controls_ = controls;
    if ( !controls_ )
    {
        auto thisControls = std::make_shared<TransformControls>();
        TransformControls::VisualParams params;
        params.update( box );
        thisControls->setVisualParams( params );
        controls_ = thisControls;
    }
    // make x - arrow
    controlsRoot_ = std::make_shared<Object>();
    controlsRoot_->setName( "TransformWidgetRoot" );
    controlsRoot_->setAncillary( true );

    controls_->setCenter( box.center() );
    controls_->init( controlsRoot_ );

    SceneRoot::get().addChild( controlsRoot_ );
    setControlsXf_( worldXf, true );

    setTransformMode( ControlBit::FullMask );

    // 10 group to imitate plugins behavior
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
    xfValidatorConnection_ = controlsRoot_->worldXfChangedSignal.connect( [&] ()
    {
        if ( !approvedChange_ && picked_ )
            stopModify_();
    } );
}

void ObjectTransformWidget::reset()
{
    if ( !controlsRoot_ )
        return;

    if ( picked_ )
        stopModify_();

    disconnect();
    xfValidatorConnection_.disconnect();

    startModifyCallback_ = {};
    stopModifyCallback_ = {};
    addXfCallback_ = {};
    scaleTooltipCallback_ = {};
    translateTooltipCallback_ = {};
    rotateTooltipCallback_ = {};

    if ( controls_ )
        controls_.reset();

    if ( controlsRoot_ )
    {
        controlsRoot_->detachFromParent();
        controlsRoot_.reset();
    }

    visibilityParent_.reset();

    axisTransformMode_ = AxisTranslation;
}

void ObjectTransformWidget::setTransformMode( ControlBit mask, ViewportId vpId )
{
    if ( !controlsRoot_ )
        return;
    if ( transformModeMask_.get( vpId ) == mask )
        return;

    transformModeMask_.set( mask, vpId );

    controls_->updateVisualTransformMode( mask,
        vpId ? vpId : ( controlsRoot_->visibilityMask() & getViewerInstance().getPresentViewports() ),
        getControlsXf( vpId ) );
}

void ObjectTransformWidget::setControlsXf( const AffineXf3f& xf, ViewportId id )
{
    setControlsXf_( xf, true, id );
}

AffineXf3f ObjectTransformWidget::getControlsXf( ViewportId id ) const
{
    return controlsRoot_->xf( id );
}

void ObjectTransformWidget::followObjVisibility( const std::weak_ptr<Object>& obj )
{
    visibilityParent_ = obj;
}

bool ObjectTransformWidget::onMouseDown_( Viewer::MouseButton button, int )
{
    if ( button != Viewer::MouseButton::Left )
        return false;
    if ( controls_->getHoveredControl() == ControlBit::None )
        return false;
    if ( !controlsRoot_ )
        return false;
    if ( !controlsRoot_->globalVisibility( getViewerInstance().getHoveredViewportId() ) )
        return false;

    if ( startModifyCallback_ )
        startModifyCallback_();

    getViewerInstance().select_hovered_viewport();
    picked_ = true;
    activeMove_( true );
    return true;
}

bool ObjectTransformWidget::onMouseUp_( Viewer::MouseButton button, int )
{
    if ( button != Viewer::MouseButton::Left )
        return false;
    if ( !picked_ )
        return false;
    if ( !controlsRoot_ )
        return false;

    stopModify_();

    return true;
}

bool ObjectTransformWidget::onMouseMove_( int, int )
{
    if ( !controlsRoot_ )
        return false;
    if ( !controlsRoot_->globalVisibility( getViewerInstance().getHoveredViewportId() ) )
        return false;
    if ( picked_ )
        activeMove_();
    else
        controls_->hover();
    return picked_;
}

void ObjectTransformWidget::preDraw_()
{
    if ( !controlsRoot_ )
        return;
    if ( auto parent = visibilityParent_.lock() )
        controlsRoot_->setVisibilityMask( parent->globalVisibilityMask() );
    auto vpmask = controlsRoot_->visibilityMask() & getViewerInstance().getPresentViewports();
    for ( auto vpId : vpmask )
    {
        auto showMask = transformModeMask_.get( vpId );
        controls_->updateVisualTransformMode( showMask, vpId, getControlsXf( vpId ) );
        controls_->updateSizeInPixel();
    }
}

void ObjectTransformWidget::postDraw_()
{
    if ( !picked_ )
        return;

    switch ( activeEditMode_ )
    {
    case TranslationMode:
        if ( translateTooltipCallback_ )
            translateTooltipCallback_( accumShift_ );
        break;
    case ScalingMode:
    case UniformScalingMode:
        if ( scaleTooltipCallback_ )
            scaleTooltipCallback_( currentScaling_ );
        break;
    case RotationMode:
        if ( rotateTooltipCallback_ )
            rotateTooltipCallback_( accumAngle_ );
        break;
    }
}

void ObjectTransformWidget::activeMove_( bool press )
{
    auto activeControl = controls_->getHoveredControl();
    assert( activeControl != ControlBit::None );

    if ( press )
    {
        // we now know who is picked
        if ( bool( activeControl & ControlBit::MoveMask ) )
        {
            switch ( axisTransformMode_ )
            {
            case AxisTranslation:
                activeEditMode_ = TranslationMode;
                break;
            case AxisScaling:
                activeEditMode_ = ScalingMode;
                break;
            case UniformScaling:
                activeEditMode_ = UniformScalingMode;
                break;
            }
        }
        else
        {
            activeEditMode_ = RotationMode;
        }
    }
    Axis activeAxis;
    switch ( activeControl )
    {
    case MR::ControlBit::MoveX:
    case MR::ControlBit::RotX:
        activeAxis = Axis::X;
        break;
    case MR::ControlBit::MoveY:
    case MR::ControlBit::RotY:
        activeAxis = Axis::Y;
        break;
    case MR::ControlBit::MoveZ:
    case MR::ControlBit::RotZ:
        activeAxis = Axis::Z;
        break;
    default:
        activeAxis = Axis::X;
        break;
    }


    switch ( activeEditMode_ )
    {
    case TranslationMode:
        processTranslation_( activeAxis, press );
        break;
    case ScalingMode:
    case UniformScalingMode:
        processScaling_( activeAxis, press );
        break;
    case RotationMode:
        processRotation_( activeAxis, press );
        break;
    }
}

void ObjectTransformWidget::processScaling_( Axis ax, bool press )
{
    const auto& mousePos = getViewerInstance().mouseController().getMousePos();
    auto& viewport = getViewerInstance().viewport();
    auto viewportPoint = getViewerInstance().screenToViewport( Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f ), viewport.id );
    auto line = viewport.unprojectPixelRay( Vector2f( viewportPoint.x, viewportPoint.y ) );
    auto xf = controlsRoot_->xf( viewport.id );
    const auto& wCenter = controls_->getCenter();
    auto newScaling = closestPoints( Line3f( xf( wCenter ), xf.A * baseAxis[int( ax )] ), line ).a;
    auto centerTransformed = xf( controls_->getCenter() );

    if ( press )
    {
        prevScaling_ = newScaling;
        currentScaling_ = 1.0f;
    }

    auto scaleFactor = ( newScaling - centerTransformed ).length() / ( prevScaling_ - centerTransformed ).length();
    currentScaling_ *= scaleFactor;
    prevScaling_ = newScaling;

    if ( activeEditMode_ == UniformScalingMode )
    {
        auto uniScale = Vector3f::diagonal( scaleFactor );
        auto uniScaleXf = AffineXf3f::xfAround( Matrix3f::scale( uniScale ), centerTransformed );
        addXf_( uniScaleXf );
    }
    else if ( activeEditMode_ == ScalingMode )
    {
        auto scale = Vector3f::diagonal( 1.f );
        scale[int( ax )] = scaleFactor;
        auto addXf = xf * AffineXf3f::xfAround( Matrix3f::scale( scale ), controls_->getCenter() ) * xf.inverse();
        addXf_( addXf );
    }
    else
        assert( false );
}

void ObjectTransformWidget::processTranslation_( Axis ax, bool press )
{
    const auto& mousePos = getViewerInstance().mouseController().getMousePos();
    auto& viewport = getViewerInstance().viewport();
    auto viewportPoint = getViewerInstance().screenToViewport( Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f ), viewport.id );
    auto line = viewport.unprojectPixelRay( Vector2f( viewportPoint.x, viewportPoint.y ) );
    auto xf = controlsRoot_->xf( viewport.id );
    const auto& wCenter = controls_->getCenter();
    auto newTranslation = closestPoints( Line3f( xf( wCenter ), xf.A * baseAxis[int( ax )] ), line ).a;

    if ( press )
    {
        accumShift_ = 0.0f;
        prevTranslation_ = startTranslation_ = newTranslation;
    }

    auto addXf = AffineXf3f::translation( newTranslation - prevTranslation_ );
    addXf_( addXf );
    prevTranslation_ = newTranslation;

    accumShift_ = dot( newTranslation - startTranslation_, ( xf.A * baseAxis[int( ax )] ).normalized() );

    if ( controls_ )
        controls_->updateTranslation( ax, startTranslation_, newTranslation, viewport.id );
}

void ObjectTransformWidget::processRotation_( Axis ax, bool press )
{
    const auto& mousePos = getViewerInstance().mouseController().getMousePos();
    auto& viewport = getViewerInstance().viewport();
    auto viewportPoint = getViewerInstance().screenToViewport( Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f ), viewport.id );
    auto line = viewport.unprojectPixelRay( Vector2f( viewportPoint.x, viewportPoint.y ) );

    auto xf = controlsRoot_->xf( viewport.id );
    const auto& wCenter = controls_->getCenter();
    auto wRadius = controls_->getRadius();
    auto zeroPoint = xf( wCenter + baseAxis[( int( ax ) + 1 ) % 3] * wRadius );
    auto norm = xf.A * ( baseAxis[int( ax )] );
    auto centerTransformed = xf( wCenter );
    auto angle = findAngleDegOfPick( centerTransformed, zeroPoint, norm, line, viewport, viewportPoint );

    if ( press )
    {
        accumAngle_ = 0.0f;
        startAngle_ = angle;
    }

    auto addXf = AffineXf3f::xfAround( Matrix3f::rotation( xf.A * baseAxis[int( ax )], angle - startAngle_ ), centerTransformed );

    addXf_( addXf );
    accumAngle_ += ( angle - startAngle_ );

    if ( accumAngle_ > PI_F )
        accumAngle_ = accumAngle_ - 2.0f * PI_F;
    else if ( accumAngle_ < -PI_F )
        accumAngle_ = 2.0f * PI_F + accumAngle_;

    if ( controls_ )
        controls_->updateRotation( ax, controlsRoot_->xf( viewport.id ), startAngle_, startAngle_ + accumAngle_, viewport.id );
}

void ObjectTransformWidget::setControlsXf_( const AffineXf3f& xf, bool updateScaled, ViewportId id )
{
    if ( updateScaled )
        scaledXf_.set( xf, id );
    Matrix3f rotation, scaling;
    decomposeMatrix3( scaledXf_.get( id ).A, rotation, scaling );

    auto scaledBoxDiagonal = scaledXf_.get( id ).A * boxDiagonal_;
    float uniformScaling = scaledBoxDiagonal.length() / boxDiagonal_.length();
    Vector3f invScaling{ 1.f / scaling.x.x, 1.f / scaling.y.y, 1.f / scaling.z.z };

    approvedChange_ = true;
    controlsRoot_->setXf( scaledXf_.get( id ) * AffineXf3f::xfAround( Matrix3f::scale( invScaling ) * Matrix3f::scale( uniformScaling ), controls_->getCenter() ), id );
    approvedChange_ = false;
}

void ObjectTransformWidget::addXf_( const AffineXf3f& addXf )
{
    if ( addXf == AffineXf3f() )
        return;

    if ( approveXfCallback_ && !approveXfCallback_( addXf ) )
        return;
    approvedChange_ = true;
    if ( addXfCallback_ )
        addXfCallback_( addXf );

    auto& defaultXf = scaledXf_.get();
    scaledXf_.set( addXf * defaultXf );
    setControlsXf_( scaledXf_.get(), false );
    for ( auto vpId : ViewportMask::all() )
    {
        bool isDef = false;
        const auto& xf = scaledXf_.get( vpId, &isDef );
        if ( isDef )
            continue;
        scaledXf_.set( addXf * xf, vpId );
        setControlsXf_( scaledXf_.get( vpId ), false, vpId );
    }

    approvedChange_ = false;
}

void ObjectTransformWidget::stopModify_()
{
    picked_ = false;

    controls_->stopModify();

    if ( stopModifyCallback_ )
        stopModifyCallback_();
}

void TransformControls::setVisualParams( const VisualParams& params )
{
    params_ = params;
    update();
}

TransformControls::~TransformControls()
{
    hoveredObject_.reset();

    for ( auto& obj : translateLines_ )
    {
        if ( !obj )
            continue;
        obj->detachFromParent();
        obj.reset();
    }

    for ( auto& obj : translateControls_ )
    {
        if ( !obj )
            continue;
        obj->detachFromParent();
        obj.reset();
    }

    for ( auto& obj : rotateLines_ )
    {
        if ( !obj )
            continue;
        obj->detachFromParent();
        obj.reset();
    }

    for ( auto& obj : rotateControls_ )
    {
        if ( !obj )
            continue;
        obj->detachFromParent();
        obj.reset();
    }

    if ( activeLine_ )
    {
        activeLine_->detachFromParent();
        activeLine_.reset();
    }
}

void TransformControls::init( std::shared_ptr<Object> parent )
{
    float radius = params_.sizeType == VisualParams::SizeType::LengthUnit ? params_.radius : 1.0f;
    float width = params_.sizeType == VisualParams::SizeType::LengthUnit ? params_.width : params_.width / params_.radius;
    for ( int i = int( Axis::X ); i < int( Axis::Count ); ++i )
    {
        if ( !translateControls_[i] )
        {
            translateControls_[i] = std::make_shared<ObjectMesh>();
            translateControls_[i]->setAncillary( true );
            translateControls_[i]->setFlatShading( true );
            translateControls_[i]->setName( "TranslationC " + std::to_string( i ) );
            if ( parent )
                parent->addChild( translateControls_[i] );
        }
        translateControls_[i]->setFrontColor( params_.translationColors[i], false );
        if ( !translateLines_[i] )
        {
            translateLines_[i] = std::make_shared<ObjectLines>();
            translateLines_[i]->setAncillary( true );
            translateLines_[i]->setVisualizeProperty( false, VisualizeMaskType::DepthTest, ViewportMask::all() );
            translateLines_[i]->setName( "TranslationL " + std::to_string( i ) );
            translateControls_[i]->addChild( translateLines_[i] );
        }
        translateLines_[i]->setFrontColor( params_.helperLineColor, false );
        auto transPolyline = std::make_shared<Polyline3>();
        std::vector<Vector3f> translationPoints =
        {
            getCenter() - radius * params_.negativeLineExtension * baseAxis[i],
            getCenter() + radius * params_.positiveLineExtension * baseAxis[i]
        };
        transPolyline->addFromPoints( translationPoints.data(), translationPoints.size() );
        translateLines_[i]->setPolyline( transPolyline );

        translateControls_[i]->setMesh( std::make_shared<Mesh>(
            makeArrow( translationPoints[0], translationPoints[1], width, params_.coneRadiusFactor * width, params_.coneSizeFactor * width ) ) );

        auto xf = AffineXf3f::translation( getCenter() ) *
            AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), baseAxis[i] ) );

        if ( !rotateControls_[i] )
        {
            rotateControls_[i] = std::make_shared<ObjectMesh>();
            rotateControls_[i]->setAncillary( true );
            rotateControls_[i]->setFlatShading( true );
            rotateControls_[i]->setName( "RotationC " + std::to_string( i ) );
            if ( parent )
                parent->addChild( rotateControls_[i] );
        }
        rotateControls_[i]->setFrontColor( params_.rotationColors[i], false );
        if ( !rotateLines_[i] )
        {
            rotateLines_[i] = std::make_shared<ObjectLines>();
            rotateLines_[i]->setAncillary( true );
            rotateLines_[i]->setVisualizeProperty( false, VisualizeMaskType::DepthTest, ViewportMask::all() );
            rotateLines_[i]->setName( "RotationL " + std::to_string( i ) );
            rotateControls_[i]->addChild( rotateLines_[i] );
        }
        rotateLines_[i]->setFrontColor( params_.helperLineColor, false );
        auto rotPolyline = std::make_shared<Polyline3>();
        std::vector<Vector3f> rotatePoints;
        auto rotMesh = makeTorus( radius, width, 128, 32, &rotatePoints );
        for ( auto& p : rotatePoints )
            p = xf( p );
        rotPolyline->addFromPoints( rotatePoints.data(), rotatePoints.size(), true );
        rotateLines_[i]->setPolyline( rotPolyline );

        rotMesh.transform( xf );
        rotateControls_[i]->setMesh( std::make_shared<Mesh>( std::move( rotMesh ) ) );
    }

    if ( !activeLine_ )
    {
        activeLine_ = std::make_shared<ObjectLines>();
        activeLine_->setVisible( false );
        activeLine_->setAncillary( true );
        activeLine_->setFrontColor( params_.activeLineColor, false );
        activeLine_->setLineWidth( 3.0f );
        activeLine_->setVisualizeProperty( false, VisualizeMaskType::DepthTest, ViewportMask::all() );
        activeLine_->setName( "Active line" );
        SceneRoot::get().addChild( activeLine_ );
    }
}

void TransformControls::update()
{
    if ( translateControls_[0] )
        init( {} );
}

void TransformControls::setRadius( float radius )
{
    if ( params_.radius == radius )
        return;
    params_.radius = radius;

    update();
}

void TransformControls::setWidth( float width )
{
    if ( params_.width == width )
        return;
    params_.width = width;

    update();
}

void TransformControls::setSizeType( VisualParams::SizeType type )
{
    if ( params_.sizeType == type )
        return;

    resetSizeInPixel_();

    params_.sizeType = type;
}

void TransformControls::updateSizeInPixel()
{
    if ( params_.sizeType != VisualParams::SizeType::Pixels )
        return;

    if ( !translateControls_[0] )
        return;

    auto parent = translateControls_[0]->parent();
    if ( !parent )
        return;

    auto mask = getViewerInstance().getPresentViewports();
    for ( auto idViewport : mask )
    {
        const auto& xf = parent->worldXf(idViewport);
        const auto& center = xf( getCenter() );
        float lenPerPixel = getViewerInstance().viewport( idViewport ).getPixelSizeAtPoint( center );

        AffineXf3f pixelTransform;

        auto radius = params_.radius * lenPerPixel ;
        pixelTransform = AffineXf3f::xfAround( Matrix3f::scale( radius ), getCenter() );

        for ( int i = int( Axis::X ); i < int( Axis::Count ); ++i )
        {
            translateControls_[i]->setXf( pixelTransform, idViewport );
            rotateControls_[i]->setXf( pixelTransform, idViewport );
        }
    }
}

void TransformControls::resetSizeInPixel_()
{
    for ( int i = int( Axis::X ); i < int( Axis::Count ); ++i )
    {
        if ( translateControls_[i] )
            translateControls_[i]->setXfsForAllViewports( {} );
        if ( rotateControls_[i] )
            rotateControls_[i]->setXfsForAllViewports( {} );
    }
}

ControlBit TransformControls::hover_( bool pickThrough )
{
    int hoveredInd = findHoveredIndex_();

    auto& linesArray = hoveredInd < 3 ? translateLines_ : rotateLines_;
    if ( hoveredInd > 2 )
        hoveredInd -= 3;

    auto dropCurrentObj = [&] ()
    {
        if ( hoveredObject_ )
        {
            auto color = hoveredObject_->getFrontColor( true );
            hoveredObject_->setFrontColor( color, false );
            assert( hoveredInd >= 0 );
            linesArray[hoveredInd]->setFrontColor( params_.helperLineColor, false );
            linesArray[hoveredInd]->setLineWidth( 1.0f );
        }
        hoveredObject_.reset();
        hoveredInd = -1;
    };

    std::vector<VisualObject*> objsToPick_;
    objsToPick_.reserve( 6 );
    auto hoveredViewportId = getViewerInstance().getHoveredViewportId();

    if ( pickThrough )
    {
        for ( auto obj : translateControls_ )
        {
            if ( obj->isVisible( hoveredViewportId ) )
                objsToPick_.push_back( obj.get() );
        }
        for ( auto obj : rotateControls_ )
        {
            if ( obj->isVisible( hoveredViewportId ) )
                objsToPick_.push_back( obj.get() );
        }
    }

    const auto& vp = getViewerInstance().viewport( hoveredViewportId );

    auto [obj, pick] = pickThrough ? vp.pickRenderObject( objsToPick_ ) : vp.pick_render_object();
    if ( !obj )
    {
        dropCurrentObj();
        return ControlBit::None;
    }
    auto meshObj = std::dynamic_pointer_cast< ObjectMesh >( obj );
    if ( !meshObj )
    {
        dropCurrentObj();
        return ControlBit::None;
    }
    bool isControl = meshObj->parent() == translateControls_[0]->parent();
    if ( !isControl )
    {
        dropCurrentObj();
        return ControlBit::None;
    }

    // here we picked one of controls for sure
    if ( hoveredObject_ != meshObj )
    {
        dropCurrentObj();

        hoveredObject_ = meshObj;
        auto color = hoveredObject_->getFrontColor( false );
        hoveredObject_->setFrontColor( color, true ); // save color in selected color holder
        color = 0.5f * color;
        color.a = 255;
        hoveredObject_->setFrontColor( color, false );

        if ( pickThrough )
        {
            hoveredInd = findHoveredIndex_();
            assert( hoveredInd >= 0 );
            auto& newPickLinesArray = hoveredInd < 3 ? translateLines_ : rotateLines_;
            if ( hoveredInd > 2 )
                hoveredInd -= 3;
            newPickLinesArray[hoveredInd]->setFrontColor( hoveredObject_->getFrontColor( true ), false );
            newPickLinesArray[hoveredInd]->setLineWidth( 3.0f );
        }
    }

    switch ( findHoveredIndex_() )
    {
    case 0:
        return ControlBit::MoveX;
    case 1:
        return ControlBit::MoveY;
    case 2:
        return ControlBit::MoveZ;
    case 3:
        return ControlBit::RotX;
    case 4:
        return ControlBit::RotY;
    case 5:
        return ControlBit::RotZ;
    default:
        return ControlBit::None;
    }
}

void TransformControls::stopModify_()
{
    activeLine_->setVisible( false );
    for ( auto& obj : translateLines_ )
        if ( obj )
            obj->setVisible( true );

    for ( auto& obj : rotateLines_ )
        if ( obj )
            obj->setVisible( true );
}

void TransformControls::updateVisualTransformMode_( ControlBit showMask, ViewportMask viewportMask )
{
    for ( int i = 0; i < 3; ++i )
    {
        ControlBit checkMask = ControlBit( int( ControlBit::MoveX ) << i );
        bool enable = ( showMask & checkMask ) == checkMask;
        translateControls_[i]->setVisible( enable, viewportMask );

        checkMask = ControlBit( int( ControlBit::RotX ) << i );
        enable = ( showMask & checkMask ) == checkMask;
        rotateControls_[i]->setVisible( enable, viewportMask );
    }
}

void TransformControls::updateTranslation( Axis, const Vector3f& startMove, const Vector3f& endMove, ViewportId )
{
    setActiveLineFromPoints_( { startMove,endMove } );
}

void TransformControls::updateRotation( Axis ax, const AffineXf3f& xf, float startAngle, float endAngle, ViewportId vpId )
{
    std::vector<Vector3f> activePoints;
    activePoints.reserve( 182 );
    activePoints.resize( 0 );
    endAngle = startAngle - ( endAngle - startAngle );
    int step = 1;
    if ( ( endAngle - startAngle ) < 0.0f )
        step = -1;

    auto radius = ( rotateControls_[int( ax )]->xf( vpId ).A * ( rotateLines_[0]->polyline()->points.vec_[0] - getCenter() ) ).length();
    Vector3f basisXTransfomed = xf.A * baseAxis[( int( ax ) + 1 ) % 3];
    Vector3f basisYTransfomed = xf.A * baseAxis[( int( ax ) + 2 ) % 3];

    auto centerTransformed = xf( getCenter() );

    activePoints.push_back( centerTransformed +
                            std::cos( startAngle ) * radius * basisXTransfomed +
                            std::sin( startAngle ) * radius * basisYTransfomed );

    if ( std::abs( ( endAngle - startAngle ) * 180 / PI_F ) > 1 )
        for ( int curAng = int( startAngle * 180 / PI_F + step ); curAng != int( ( endAngle ) * 180 / PI_F ); curAng += step )
        {
            activePoints.push_back( centerTransformed +
                                    std::cos( float( curAng ) * PI_F / 180 ) * radius * basisXTransfomed +
                                    std::sin( float( curAng ) * PI_F / 180 ) * radius * basisYTransfomed );
        }
    activePoints.push_back( centerTransformed +
                            std::cos( endAngle ) * radius * basisXTransfomed +
                            std::sin( endAngle ) * radius * basisYTransfomed );

    setActiveLineFromPoints_( activePoints );
}

TransformModesValidator TransformControls::ThresholdDotValidator( float thresholdDot )
{
    if ( thresholdDot <= 0.0f )
        return {};
    return [thresholdDot] ( const Vector3f& center, const AffineXf3f& xf, ViewportId vpId )
    {
        auto transformedCenter = xf( center );
        auto vpPoint = getViewerInstance().viewport( vpId ).projectToViewportSpace( transformedCenter );
        auto ray = getViewerInstance().viewport( vpId ).unprojectPixelRay( Vector2f( vpPoint.x, vpPoint.y ) ).d.normalized();

        ControlBit showMask = ControlBit::FullMask;

        bool xHide = std::abs( dot( xf.A.col( 0 ).normalized(), ray ) ) < thresholdDot;
        bool yHide = std::abs( dot( xf.A.col( 1 ).normalized(), ray ) ) < thresholdDot;
        bool zHide = std::abs( dot( xf.A.col( 2 ).normalized(), ray ) ) < thresholdDot;

        if ( xHide )
            showMask &= ~ControlBit::RotX;
        if ( yHide )
            showMask &= ~ControlBit::RotY;
        if ( zHide )
            showMask &= ~ControlBit::RotZ;

        if ( xHide && yHide )
            showMask &= ~ControlBit::MoveZ;
        if ( xHide && zHide )
            showMask &= ~ControlBit::MoveY;
        if ( yHide && zHide )
            showMask &= ~ControlBit::MoveX;

        return showMask;
    };
}

int TransformControls::findHoveredIndex_() const
{
    if ( !hoveredObject_ )
        return -1;

    // 0-2 three is translation, 3-5 is rotation
    for ( int ax = int( Axis::X ); ax < int( Axis::Count ); ++ax )
    {
        if ( hoveredObject_ == translateControls_[ax] )
            return ax;
        else if ( hoveredObject_ == rotateControls_[ax] )
            return 3 + ax;
    }
    return -1;
}

void TransformControls::setActiveLineFromPoints_( const Contour3f& points )
{
    auto activePolyline = std::make_shared<Polyline3>();
    activePolyline->addFromPoints( points.data(), points.size(), false );
    activeLine_->setPolyline( activePolyline );
    activeLine_->setVisible( true, getViewerInstance().viewport().id );
    auto disableBlackLinesIfNeeded = [&] ( auto& obj )
    {
        if ( !getPickThrough() || obj->getFrontColor( false ) == params_.helperLineColor )
            obj->setVisible( false );
        else
        {
            auto color = 0.5f * params_.helperLineColor + 0.5f * params_.activeLineColor;
            color.a = 255;
            obj->setFrontColor( color, false );
            obj->setLineWidth( 1.0f );
        }
    };
    for ( auto& obj : translateLines_ )
        disableBlackLinesIfNeeded( obj );
    for ( auto& obj : rotateLines_ )
        disableBlackLinesIfNeeded( obj );
}

void ITransformControls::setCenter( const Vector3f& center )
{
    if ( center_ == center )
        return;
    center_ = center;
    update();
}

void ITransformControls::updateVisualTransformMode( ControlBit showMask, ViewportMask viewportMask, const AffineXf3f& xf )
{
    if ( !validator_ )
    {
        updateVisualTransformMode_( showMask, viewportMask );
    }
    else
    {
        for ( auto id : viewportMask )
        {
            ControlBit modeMask = showMask & validator_( center_, xf, id );
            updateVisualTransformMode_( modeMask, id );
        }
    }
}

void TransformControls::VisualParams::update( const Box3f& box )
{
    if( radius < 0 )
        radius = box.diagonal() * 0.5f;

    if ( width < 0 )
        width = radius / 40.0f;
}

}
