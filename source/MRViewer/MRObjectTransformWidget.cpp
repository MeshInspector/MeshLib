#include "MRObjectTransformWidget.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRArrow.h"
#include "MRMesh/MRTorus.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRViewport.h"
#include "MRPch/MRTBB.h"
#include "MRMesh/MRIntersection.h"
#include "MRMesh/MR2to3.h"
#include "MRMesh/MRMatrix3Decompose.h"
#include "MRPch/MRSpdlog.h"

namespace
{
constexpr std::array<MR::Vector3f, 3> baseAxis =
{ MR::Vector3f::plusX(),MR::Vector3f::plusY(),MR::Vector3f::plusZ() };

using namespace MR;

// fins point on p11,p12 line
Vector3f findClosestPointOfSkewLines( const Vector3f& p11, const Vector3f& p12, const Vector3f& p21, const Vector3f& p22 )
{
    auto d1 = p12 - p11;
    auto d2 = p22 - p21;
    auto n = cross( d1, d2 );
    auto n2 = cross( d2, n );
    auto lSq = n.lengthSq();
    auto l2Sq = n2.lengthSq();
    if ( std::isnan( lSq ) || std::isnan( l2Sq ) )
        return {};

    return p11 + dot( ( p21 - p11 ), n2 ) / dot( d1, n2 ) * d1;
}

float findAngleDegOfPick( const Vector3f& center, const Vector3f& zeroPoint, const Vector3f& norm,
                          const Line3f& ray, Viewport& vp, const Vector3f& vpPoint )
{
    Plane3f plane1 = Plane3f::fromDirAndPt( norm, center ).normalized();
    Plane3f plane2 = Plane3f::fromDirAndPt( cross( ray.d, center - ray.p ), center ).normalized();
    auto centerVp = vp.projectToViewportSpace( center );
    auto centerRay = vp.unprojectPixelRay( to2dim( centerVp ) );
    bool parallel = std::abs( dot( centerRay.d.normalized(), norm.normalized() ) ) < 0.05f;
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

void ObjectTransformWidget::create( const Box3f& box, const AffineXf3f& worldXf )
{
    if ( controlsRoot_ )
        reset();

    center_ = box.center();
    if ( radius_ < 0.0f )
        radius_ = box.diagonal() * 0.5f;
    if ( width_ < 0.0f )
        width_ = radius_ / 40.0f;
    // make x - arrow
    controlsRoot_ = std::make_shared<Object>();
    controlsRoot_->setAncillary( true );
    
    makeControls_();

    activeLine_ = std::make_shared<ObjectLines>();
    activeLine_->setVisible( false );
    activeLine_->setAncillary( true );
    activeLine_->setFrontColor( Color::white(), false );
    activeLine_->setLineWidth( 3.0f );
    activeLine_->setVisualizeProperty( false, VisualizeMaskType::DepthTest, ViewportMask::all() );

    SceneRoot::get().addChild( controlsRoot_ );
    SceneRoot::get().addChild( activeLine_ );
    setControlsXf( worldXf );

    setTransformMode( MoveX | MoveY | MoveZ | RotX | RotY | RotZ );

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

    currentObj_.reset();

    disconnect();
    xfValidatorConnection_.disconnect();

    startModifyCallback_ = {};
    stopModifyCallback_ = {};
    addXfCallback_ = {};
    scaleTooltipCallback_ = {};
    translateTooltipCallback_ = {};
    rotateTooltipCallback_ = {};

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

    if ( controlsRoot_ )
    {
        controlsRoot_->detachFromParent();
        controlsRoot_.reset();
    }

    visibilityParent_.reset();

    width_ = -1.0f;
    radius_ = -1.0f;

    axisTransformMode_ = AxisTranslation;

    thresholdDot_ = 0.f;
}

void ObjectTransformWidget::setWidth( float width )
{
    if ( width_ == width )
        return;
    width_ = width;
    if ( radius_ > 0.0f )
        makeControls_();
}

void ObjectTransformWidget::setRadius( float radius )
{
    if ( radius == radius_ )
        return;
    radius_ = radius;
    if ( width_ > 0.0f )
        makeControls_();
}

void ObjectTransformWidget::setTransformMode( uint8_t mask )
{
    if ( !controlsRoot_ )
        return;
    if ( transformModeMask_ == mask )
        return;

    transformModeMask_ = mask;

    auto visMask = controlsRoot_->visibilityMask();
    
    updateVisualTransformMode_( transformModeMask_, visMask );
}

void ObjectTransformWidget::setControlsXf( const AffineXf3f &xf )
{
    scaledXf_ = xf;

    Matrix3f rotation, scaling;
    decomposeMatrix3( xf.A, rotation, scaling );

    Vector3f invScaling { 1.f / scaling.x.x, 1.f / scaling.y.y, 1.f / scaling.z.z };
    auto maxScaling = std::max( { scaling.x.x, scaling.y.y, scaling.z.z } );
    controlsRoot_->setXf( xf * AffineXf3f::xfAround( Matrix3f::scale( invScaling ) * Matrix3f::scale( maxScaling ), center_ ) );

    objScale_ = scaling;
}

AffineXf3f ObjectTransformWidget::getControlsXf() const
{
    return controlsRoot_->xf();
}

void ObjectTransformWidget::followObjVisibility( const std::weak_ptr<Object>& obj )
{
    visibilityParent_ = obj;
}

bool ObjectTransformWidget::onMouseDown_( Viewer::MouseButton button, int )
{
    if ( button != Viewer::MouseButton::Left )
        return false;
    if ( !currentObj_ )
        return false;
    if ( !controlsRoot_ )
        return false;
    if ( !controlsRoot_->globalVisibilty( getViewerInstance().getHoveredViewportId() ) )
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
    if ( !activeLine_ )
        return false;
    
    stopModify_();

    return true;
}

bool ObjectTransformWidget::onMouseMove_( int, int )
{
    if ( !controlsRoot_ )
        return false;
    if ( !controlsRoot_->globalVisibilty( getViewerInstance().getHoveredViewportId() ) )
        return false;
    if ( picked_ )
        activeMove_();
    else
        passiveMove_();
    return picked_;
}

void ObjectTransformWidget::preDraw_()
{
    if ( !controlsRoot_ )
        return;
    if ( auto parent = visibilityParent_.lock() )
        controlsRoot_->setVisibilityMask( parent->visibilityMask() );
    auto vpmask = controlsRoot_->visibilityMask() & getViewerInstance().getPresentViewports();
    for ( auto vpId : vpmask )
    {
        auto showMask = transformModeMask_;

        if ( thresholdDot_ > 0.0f )
        {
            const auto& xf = controlsRoot_->xf();
            auto transformedCenter = xf( center_ );
            auto vpPoint = getViewerInstance().viewport( vpId ).projectToViewportSpace( transformedCenter );
            auto ray = getViewerInstance().viewport( vpId ).unprojectPixelRay( Vector2f( vpPoint.x, vpPoint.y ) ).d.normalized();

            bool xHide = std::abs( dot( xf.A.col( 0 ).normalized(), ray ) ) < thresholdDot_;
            bool yHide = std::abs( dot( xf.A.col( 1 ).normalized(), ray ) ) < thresholdDot_;
            bool zHide = std::abs( dot( xf.A.col( 2 ).normalized(), ray ) ) < thresholdDot_;

            if ( xHide )
                showMask &= ~RotX;
            if ( yHide )
                showMask &= ~RotY;
            if ( zHide )
                showMask &= ~RotZ;

            if ( xHide && yHide )
                showMask &= ~MoveZ;
            if ( xHide && zHide )
                showMask &= ~MoveY;
            if ( yHide && zHide )
                showMask &= ~MoveX;
        }
        updateVisualTransformMode_( showMask, vpId );
    }
}

void ObjectTransformWidget::draw_()
{
    if ( !picked_ )
        return;
    int currentIndex = findCurrentObjIndex_();
    if ( currentIndex < 0 )
        return;

    switch ( activeEditMode_ )
    {
    case TranslationMode:
        if ( translateTooltipCallback_ )
        {
            auto xf = controlsRoot_->xf();
            auto axis = xf( translateLines_[currentIndex]->polyline()->points.vec_[1] ) -
                        xf( translateLines_[currentIndex]->polyline()->points.vec_[0] );
            translateTooltipCallback_( dot( prevTranslation_ - startTranslation_, axis.normalized() ) );
        }
        break;
    case ScalingMode:
    case UniformScalingMode:
        if ( scaleTooltipCallback_ )
            scaleTooltipCallback_( objScale_[currentIndex][currentIndex] );
        break;
    case RotationMode:
        if ( rotateTooltipCallback_ )
            rotateTooltipCallback_( accumAngle_ );
        break;
    }
}

int ObjectTransformWidget::findCurrentObjIndex_() const
{
    if ( !currentObj_ )
        return -1;

    int currnetIndex = -1;// 0-2 three is translation, 3-5 is rotation
    for ( int ax = Axis::X; ax < Axis::Count; ++ax )
        if ( currentObj_ == translateControls_[ax] )
        {
            currnetIndex = ax;
            break;
        }
    if ( currnetIndex == -1 )
        for ( int ax = Axis::X; ax < Axis::Count; ++ax )
            if ( currentObj_ == rotateControls_[ax] )
            {
                currnetIndex = 3 + ax;
                break;
            }
    assert( currnetIndex >= 0 && currnetIndex < 6 );

    return currnetIndex;
}

void ObjectTransformWidget::makeControls_()
{
    if ( !controlsRoot_ )
        return;

    for ( int i = Axis::X; i < Axis::Count; ++i )
    {
        if ( !translateControls_[i] )
        {
            translateControls_[i] = std::make_shared<ObjectMesh>();
            translateControls_[i]->setAncillary( true );
            translateControls_[i]->setFrontColor( Color( baseAxis[i] ), false );
            translateControls_[i]->setFlatShading( true );
            controlsRoot_->addChild( translateControls_[i] );
        }
        if ( !translateLines_[i] )
        {
            translateLines_[i] = std::make_shared<ObjectLines>();
            translateLines_[i]->setAncillary( true );
            translateLines_[i]->setFrontColor( Color::black(), false );
            translateLines_[i]->setVisualizeProperty( false, VisualizeMaskType::DepthTest, ViewportMask::all() );
            translateControls_[i]->addChild( translateLines_[i] );
        }
        auto transPolyline = std::make_shared<Polyline3>();
        std::vector<Vector3f> translationPoints = { center_ - radius_ * 1.15f * baseAxis[i],center_ + radius_ * 1.3f * baseAxis[i] };
        transPolyline->addFromPoints( translationPoints.data(), translationPoints.size() );
        translateLines_[i]->setPolyline( transPolyline );

        translateControls_[i]->setMesh( std::make_shared<Mesh>(
            makeArrow( translationPoints[0], translationPoints[1], width_, 1.35f * width_, 2.2f * width_ ) ) );

        auto xf = AffineXf3f::translation( center_ ) *
            AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), baseAxis[i] ) );

        if ( !rotateControls_[i] )
        {
            rotateControls_[i] = std::make_shared<ObjectMesh>();
            rotateControls_[i]->setAncillary( true );
            rotateControls_[i]->setFrontColor( Color( baseAxis[i] ), false );
            rotateControls_[i]->setFlatShading( true );
            controlsRoot_->addChild( rotateControls_[i] );
        }
        if ( !rotateLines_[i] )
        {
            rotateLines_[i] = std::make_shared<ObjectLines>();
            rotateLines_[i]->setAncillary( true );
            rotateLines_[i]->setFrontColor( Color::black(), false );
            rotateLines_[i]->setVisualizeProperty( false, VisualizeMaskType::DepthTest, ViewportMask::all() );
            rotateControls_[i]->addChild( rotateLines_[i] );
        }
        auto rotPolyline = std::make_shared<Polyline3>();
        std::vector<Vector3f> rotatePoints;
        auto rotMesh = makeTorus( radius_, width_, 128, 32, &rotatePoints );
        for ( auto& p : rotatePoints )
            p = xf( p );
        rotPolyline->addFromPoints( rotatePoints.data(), rotatePoints.size(), true );
        rotateLines_[i]->setPolyline( rotPolyline );

        rotMesh.transform( xf );
        rotateControls_[i]->setMesh( std::make_shared<Mesh>( std::move( rotMesh ) ) );
    }
}

void ObjectTransformWidget::passiveMove_()
{
    int currentIndex = findCurrentObjIndex_();
    auto& linesArray = currentIndex < 3 ? translateLines_ : rotateLines_;
    if ( currentIndex > 2 )
        currentIndex -= 3;

    auto dropCurrentObj = [&] ()
    {
        if ( currentObj_ )
        {
            auto color = currentObj_->getFrontColor( true );
            currentObj_->setFrontColor( color, false );
            assert( currentIndex >= 0 );
            linesArray[currentIndex]->setFrontColor( Color::black(), false );
            linesArray[currentIndex]->setLineWidth( 1.0f );
        }
        currentObj_.reset();
        currentIndex = -1;
    };

    std::vector<VisualObject*> objsToPick_;
    objsToPick_.reserve( 6 );
    auto hoveredViewportId = getViewerInstance().getHoveredViewportId();

    if ( pickThrough_ )
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

    auto [obj, pick] = pickThrough_ ? vp.pick_render_object( objsToPick_ ) : vp.pick_render_object();
    if ( !obj )
    {
        dropCurrentObj();
        return;
    }
    auto meshObj = std::dynamic_pointer_cast< ObjectMesh >( obj );
    if ( !meshObj )
    {
        dropCurrentObj();
        return;
    }
    bool isControl = meshObj->parent() == controlsRoot_.get();
    if ( !isControl )
    {
        dropCurrentObj();
        return;
    }

    // here we picked one of controls for sure
    if ( currentObj_ != meshObj )
    {
        dropCurrentObj();

        currentObj_ = meshObj;
        auto color = currentObj_->getFrontColor( false );
        currentObj_->setFrontColor( color, true ); // save color in selected color holder
        color = 0.5f * color;
        color.a = 255;
        currentObj_->setFrontColor( color, false );

        if ( pickThrough_ )
        {
            currentIndex = findCurrentObjIndex_();
            assert( currentIndex >= 0 );
            auto& newPickLinesArray = currentIndex < 3 ? translateLines_ : rotateLines_;
            if ( currentIndex > 2 )
                currentIndex -= 3;
            newPickLinesArray[currentIndex]->setFrontColor( currentObj_->getFrontColor( true ), false );
            newPickLinesArray[currentIndex]->setLineWidth( 3.0f );
        }
    }    
}

void ObjectTransformWidget::activeMove_( bool press )
{
    assert( currentObj_ );

    int currentObjIndex = findCurrentObjIndex_();

    if ( press )
    {
        // we now know who is picked
        if ( currentObjIndex < 3 )
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

    switch ( activeEditMode_ )
    {
    case TranslationMode:
        processTranslation_( Axis( currentObjIndex ), press );
        break;
    case ScalingMode:
    case UniformScalingMode:
        processScaling_( Axis( currentObjIndex ), press );
        break;
    case RotationMode:
        processRotation_( Axis( currentObjIndex - 3 ), press );
        break;
    }
}

void ObjectTransformWidget::processScaling_( ObjectTransformWidget::Axis ax, bool press )
{
    const auto& mousePos = getViewerInstance().mouseController.getMousePos();
    auto& viewport = getViewerInstance().viewport();
    auto viewportPoint = getViewerInstance().screenToViewport( Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f ), viewport.id );
    auto line = viewport.unprojectPixelRay( Vector2f( viewportPoint.x, viewportPoint.y ) );
    auto xf = controlsRoot_->xf();
    auto newScaling = findClosestPointOfSkewLines(
        xf( translateLines_[int( ax )]->polyline()->points.vec_[0] ),
        xf( translateLines_[int( ax )]->polyline()->points.vec_[1] ),
        line.p, line.p + line.d
    );

    if ( press )
        prevScaling_ = newScaling;

    auto scaleFactor = ( newScaling - xf( center_ ) ).length() / ( prevScaling_ - xf( center_ ) ).length();
    auto scale = Vector3f::diagonal( 1.f );
    if ( activeEditMode_ == UniformScalingMode )
        scale *= scaleFactor;
    else
        scale[int( ax )] = scaleFactor;
    prevScaling_ = newScaling;

    objScale_ = Matrix3f::scale( scale ) * objScale_;

    auto addXf = xf * AffineXf3f::xfAround( Matrix3f::scale( scale ), center_ ) * xf.inverse();
    addXf_( addXf );
}

void ObjectTransformWidget::processTranslation_( Axis ax, bool press )
{
    const auto& mousePos = getViewerInstance().mouseController.getMousePos();
    auto& viewport = getViewerInstance().viewport();
    auto viewportPoint = getViewerInstance().screenToViewport( Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f ), viewport.id );
    auto line = viewport.unprojectPixelRay( Vector2f( viewportPoint.x, viewportPoint.y ) );
    auto xf = controlsRoot_->xf();
    auto newTranslation = findClosestPointOfSkewLines(
        xf( translateLines_[int( ax )]->polyline()->points.vec_[0] ),
        xf( translateLines_[int( ax )]->polyline()->points.vec_[1] ),
        line.p, line.p + line.d
    );

    if ( press )
        prevTranslation_ = startTranslation_ = newTranslation;

    std::vector<Vector3f> activePoints( 2 );
    activePoints[0] = startTranslation_;
    activePoints[1] = newTranslation;

    auto addXf = AffineXf3f::translation( newTranslation - prevTranslation_ );
    addXf_( addXf );
    prevTranslation_ = newTranslation;

    setActiveLineFromPoints_( activePoints );
}

void ObjectTransformWidget::processRotation_( Axis ax, bool press )
{
    const auto& mousePos = getViewerInstance().mouseController.getMousePos();
    auto& viewport = getViewerInstance().viewport();
    auto viewportPoint = getViewerInstance().screenToViewport( Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f ), viewport.id );
    auto line = viewport.unprojectPixelRay( Vector2f( viewportPoint.x, viewportPoint.y ) );

    auto prevXf = controlsRoot_->xf();
    auto zeroPoint = prevXf( rotateLines_[int( ax )]->polyline()->points.vec_[0] );
    auto norm = prevXf.A * translateLines_[int( ax )]->polyline()->edgeVector( 0_e );
    auto centerTransformed = prevXf( center_ );
    auto angle = findAngleDegOfPick( centerTransformed, zeroPoint, norm, line, viewport, viewportPoint );

    if ( press )
    {
        accumAngle_ = 0.0f;
        startAngle_ = angle;
    }

    if ( press )
        startRotXf_ = prevXf;
    auto addXf =
        AffineXf3f::translation( centerTransformed ) *
        AffineXf3f::linear( Matrix3f::rotation( prevXf.A * baseAxis[ax], angle - startAngle_ ) ) *
        AffineXf3f::translation( -centerTransformed );

    addXf_( addXf );
    accumAngle_ += ( angle - startAngle_ );

    if ( accumAngle_ > PI_F )
        accumAngle_ = accumAngle_ - 2.0f * PI_F;
    else if ( accumAngle_ < -PI_F )
        accumAngle_ = 2.0f * PI_F + accumAngle_;

    std::vector<Vector3f> activePoints;
    activePoints.reserve( 182 );
    activePoints.resize( 0 );
    int step = 1;
    if ( accumAngle_ < 0.0f )
        step = -1;

    auto radius = ( rotateLines_[0]->polyline()->points.vec_[0] - center_ ).length();
    Vector3f basisXTransfomed;
    Vector3f basisYTransfomed;
    if ( ax == X )
    {
        basisXTransfomed = -( startRotXf_.A * baseAxis[2].normalized() );
        basisYTransfomed = ( startRotXf_.A * baseAxis[1].normalized() );
    }
    else if ( ax == Y )
    {
        basisXTransfomed = startRotXf_.A * baseAxis[0].normalized();
        basisYTransfomed = -( startRotXf_.A * baseAxis[2].normalized() );
    }
    else
    {
        basisXTransfomed = startRotXf_.A * baseAxis[0].normalized();
        basisYTransfomed = startRotXf_.A * baseAxis[1].normalized();
    }
    activePoints.push_back( centerTransformed +
                            std::cos( startAngle_ ) * radius * basisXTransfomed +
                            std::sin( startAngle_ ) * radius * basisYTransfomed );

    if ( std::abs( accumAngle_ * 180 / PI_F ) > 1 )
        for ( int curAng = int( startAngle_ * 180 / PI_F + step ); curAng != int( ( startAngle_ + accumAngle_ ) * 180 / PI_F ); curAng += step )
        {
            activePoints.push_back( centerTransformed +
                                    std::cos( float( curAng ) * PI_F / 180 ) * radius * basisXTransfomed +
                                    std::sin( float( curAng ) * PI_F / 180 ) * radius * basisYTransfomed );
        }
    activePoints.push_back( centerTransformed +
                            std::cos( startAngle_ + accumAngle_ ) * radius * basisXTransfomed +
                            std::sin( startAngle_ + accumAngle_ ) * radius * basisYTransfomed );

    setActiveLineFromPoints_( activePoints );
}

void ObjectTransformWidget::updateVisualTransformMode_( uint8_t showMask, ViewportMask viewportMask )
{
    for ( int i = 0; i < 3; ++i )
    {
        uint8_t checkMask = TransformMode::MoveX << i;
        bool enable = ( showMask & checkMask ) == checkMask;
        translateControls_[i]->setVisible( enable, viewportMask );

        checkMask = TransformMode::RotX << i;
        enable = ( showMask & checkMask ) == checkMask;
        rotateControls_[i]->setVisible( enable, viewportMask );
    }
}

void ObjectTransformWidget::setActiveLineFromPoints_( const std::vector<Vector3f>& points )
{
    auto activePolyline = std::make_shared<Polyline3>();
    activePolyline->addFromPoints( points.data(), points.size(), false );
    activeLine_->setPolyline( activePolyline );
    activeLine_->setVisible( true, getViewerInstance().viewport().id );
    auto disableBlackLinesIfNeeded = [&] ( auto& obj )
    {
        if ( !pickThrough_ || obj->getFrontColor( false ) == Color::black() )
            obj->setVisible( false );
        else
        {
            obj->setFrontColor( Color::gray(), false );
            obj->setLineWidth( 1.0f );
        }
    };
    for ( auto& obj : translateLines_ )
        disableBlackLinesIfNeeded( obj );
    for ( auto& obj : rotateLines_ )
        disableBlackLinesIfNeeded( obj );
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
    if ( activeEditMode_ == TranslationMode || activeEditMode_ == RotationMode )
        setControlsXf( addXf * controlsRoot_->xf() );
    scaledXf_ = addXf * scaledXf_;
    approvedChange_ = false;
}

void ObjectTransformWidget::stopModify_()
{
    picked_ = false;
    activeLine_->setVisible( false );
    for ( auto& obj : translateLines_ )
        if ( obj )
            obj->setVisible( true );

    for ( auto& obj : rotateLines_ )
        if ( obj )
            obj->setVisible( true );

    passiveMove_();

    if ( activeEditMode_ == ScalingMode || activeEditMode_ == UniformScalingMode )
        setControlsXf( scaledXf_ );

    if ( stopModifyCallback_ )
        stopModifyCallback_();
}

}