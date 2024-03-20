#include "MRSurfacePointPicker.h"
#include "MRViewport.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRSphereObject.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRRingIterator.h"

#include "MRMesh/MRObjectPointsHolder.h"
#include "MRMesh/MRObjectLines.h"

#include "MRMesh/MRMeshTriPoint.h"
#include "MRMesh/MREdgePoint.h"
#include "MRMesh/MRPointOnObject.h"
#include "MRMesh/MRPointCloud.h"

#include <variant>

namespace MR
{

SurfacePointWidget::~SurfacePointWidget()
{
    reset();
}

const PickedPoint& SurfacePointWidget::create( const std::shared_ptr<VisualObject>& surface, const PointOnObject& startPos )
{
    if ( !surface )
    {
        currentPos_ = -1;
        return currentPos_;
    }
    return create( surface, pointOnObjectToPickedPoint( baseObject_.get(), startPos ) );
}

const PickedPoint& SurfacePointWidget::create( const std::shared_ptr<VisualObject>& surface, const PickedPoint& startPos )
{
    reset();
    if ( !surface )
    {
        currentPos_ = -1;
        return currentPos_;
    }
    baseObject_ = surface;

    pickSphere_ = std::make_shared<SphereObject>();
    pickSphere_->setName( "Pick Sphere" );
    pickSphere_->setAncillary( true );
    pickSphere_->setFrontColor( params_.baseColor, false );
    pickSphere_->setBackColor( pickSphere_->getFrontColor( false ) );
    pickSphere_->setGlobalAlpha( 255 );
    pickSphere_->setVisualizeProperty( false, DimensionsVisualizePropertyType::diameter, ViewportMask::all() );
    pickSphere_->setDecorationsColor( Color::transparent(), false );

    baseObject_->addChild( pickSphere_ );
    currentPos_ = startPos;
    updatePositionAndRadius_();

    // 10 group to imitate plugins behavior
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
    return currentPos_;
}


void SurfacePointWidget::reset()
{
    if ( !pickSphere_ )
        return;

    disconnect();
    pickSphere_->detachFromParent();
    pickSphere_.reset();

    baseObject_.reset();

    params_ = Parameters();
    isOnMove_ = false;
    isHovered_ = false;
    autoHover_ = true;
    startMove_ = {};
    onMove_ = {};
    endMove_ = {};
}

void SurfacePointWidget::setParameters( const Parameters& params )
{
    if ( pickSphere_ )
    {
        if ( isHovered_ )
            pickSphere_->setFrontColor( params.hoveredColor, false );
        else if ( isOnMove_ )
            pickSphere_->setFrontColor( params.activeColor, false );
        else
            pickSphere_->setFrontColor( params.baseColor, false );

        pickSphere_->setBackColor( pickSphere_->getFrontColor( false ) );

        if ( params.positionType != params_.positionType ||
             params.radius != params_.radius )
        {
            updatePositionAndRadius_();
        }
    }
    params_ = params;
}

void SurfacePointWidget::setHovered( bool on )
{
    if ( !isOnMove_ && isHovered_ != on )
    {
        isHovered_ = on;
        pickSphere_->setFrontColor( isHovered_ ? params_.hoveredColor : params_.baseColor, false );
        pickSphere_->setBackColor( pickSphere_->getFrontColor( false ) );
    }
}

bool SurfacePointWidget::onMouseDown_( Viewer::MouseButton button, int mod )
{
    if ( button != MouseButton::Left || !isHovered_ )
        return false;

    // check if modifier present and if there are exception for it.
    if ( ( mod != 0 ) && ( ( mod & params_.customModifiers ) != mod ) )
        return false;

    pickSphere_->setPickable( false );
    isOnMove_ = true;
    pickSphere_->setFrontColor( params_.activeColor, false );
    pickSphere_->setBackColor( pickSphere_->getFrontColor( false ) );
    if ( startMove_ )
        startMove_( currentPos_ );
    return true;
}

bool SurfacePointWidget::onMouseUp_( Viewer::MouseButton button, int )
{
    if ( button != MouseButton::Left || !isOnMove_ )
        return false;
    isOnMove_ = false;
    pickSphere_->setPickable( true );
    pickSphere_->setFrontColor( params_.baseColor, false );
    pickSphere_->setBackColor( pickSphere_->getFrontColor( false ) );
    if ( endMove_ )
        endMove_( currentPos_ );
    return true;
}

bool SurfacePointWidget::onMouseMove_( int, int )
{
    if ( isOnMove_ )
    {
        auto [obj, pick] = getViewerInstance().viewport().pick_render_object( params_.pickInBackFaceObject );
        if ( obj != baseObject_ )
            return false;

        if ( ( params_.pickInBackFaceObject == false ) && ( isPickIntoBackFace( obj, pick, getViewerInstance().viewport().getCameraPoint() ) ) )
            return false;

        currentPos_ = pointOnObjectToPickedPoint( obj.get(), pick );
        updatePositionAndRadius_();
        if ( onMove_ )
            onMove_( currentPos_ );
        return true;
    }
    else
    {
        if ( !autoHover_ )
            return false;

        auto [obj, pick] = getViewerInstance().viewport().pick_render_object();
        setHovered( obj == pickSphere_ );
    }
    return false;
}

MR::Vector3f SurfacePointWidget::toVector3f() const
{
    return pickedPointToVector3( baseObject_.get(), currentPos_ );
}


void SurfacePointWidget::updatePositionAndRadiusPoints_( const VertId& /* v */ )
{
    pickSphere_->setCenter( toVector3f() );
    setPointRadius_();
}

void SurfacePointWidget::updatePositionAndRadiusLines_( const EdgePoint& /* ep */ )
{
    pickSphere_->setCenter( toVector3f() );
    setPointRadius_();
}


void SurfacePointWidget::updatePositionAndRadiusMesh_( MeshTriPoint mtp )
{
    assert( pickSphere_ );
    auto baseSurface = std::dynamic_pointer_cast< ObjectMeshHolder >( baseObject_ );
    assert( baseSurface );
    assert( baseSurface->mesh() );
    const auto& mesh = *baseSurface->mesh();
    FaceId fId = mesh.topology.left( mtp.e );
    switch ( params_.positionType )
    {
    case PositionType::FaceCenters:
    {
        currentPos_ = mesh.toTriPoint( fId, mesh.triCenter( fId ) );
        break;
    }
    case PositionType::Edges:
    {
        if ( !mtp.onEdge( mesh.topology ) )
        {
            auto closestEdge = EdgeId( mesh.getClosestEdge( PointOnFace{ fId,mesh.triPoint( mtp ) } ) );
            if ( mesh.topology.left( closestEdge ) != fId )
                closestEdge = closestEdge.sym();
            if ( mtp.e == closestEdge )
                mtp.bary.b = 0.0f;
            else if ( mtp.e == mesh.topology.next( closestEdge ).sym() )
            {
                mtp.e = closestEdge;
                mtp.bary.a = mtp.bary.b;
                mtp.bary.b = 0.0f;
            }
            else
            {
                mtp.e = closestEdge;
                mtp.bary.a = 1.0f - mtp.bary.b;
                mtp.bary.b = 0.0f;
            }
            currentPos_ = mtp;
        }
        break;
    }
    case PositionType::EdgeCeneters:
    {
        auto closestEdge = EdgeId( mesh.getClosestEdge( PointOnFace{ fId,mesh.triPoint( mtp ) } ) );
        if ( mesh.topology.left( closestEdge ) != fId )
            closestEdge = closestEdge.sym();
        mtp.e = closestEdge;
        mtp.bary.a = 0.5f;
        mtp.bary.b = 0.0f;
        currentPos_ = mtp;
        break;
    }
    case PositionType::Verts:
    {
        if ( !mtp.inVertex() )
        {
            auto closestVert = mesh.getClosestVertex( PointOnFace{ fId,mesh.triPoint( mtp ) } );
            for ( auto e : orgRing( mesh.topology, closestVert ) )
            {
                if ( mesh.topology.left( e ) == fId )
                {
                    mtp.e = e;
                    mtp.bary.a = 0.0f;
                    mtp.bary.b = 0.0f;
                    currentPos_ = mtp;
                    break;
                }
            }
        }
        break;
    }
    default:
        break;
    }

    pickSphere_->setCenter( toVector3f() );
    setPointRadius_();
}

void SurfacePointWidget::updatePositionAndRadius_()
{
    if ( const MeshTriPoint* triPoint = std::get_if<MeshTriPoint>( &currentPos_ ) )
    {
        updatePositionAndRadiusMesh_( *triPoint );
    }
    else if ( const EdgePoint* edgePoint = std::get_if<EdgePoint>( &currentPos_ ) )
    {
        updatePositionAndRadiusLines_( *edgePoint );
    }
    else if ( const VertId* vertId = std::get_if<VertId>( &currentPos_ ) )
    {
        updatePositionAndRadiusPoints_( *vertId );
    }
    else if ( std::get_if<int>( &currentPos_ ) )
    {
        return; // pick in empty space
    }
}


void SurfacePointWidget::setPointRadius_()
{
    float radius = 0;
    if ( params_.radiusSizeType == SurfacePointWidget::Parameters::PointSizeType::Pixel )
    {
        const auto& vParams = Viewer::instanceRef().viewport().getParameters();
        auto w = MR::height( Viewer::instanceRef().viewport().getViewportRect() );
        auto scale = tan( vParams.cameraViewAngle / 360.0f * PI_F ) / vParams.cameraZoom / w;
        radius = params_.radius * scale;
    }
    else
    {
        radius = params_.radius <= 0.0f ? baseObject_->getBoundingBox().diagonal() * 5e-3f : params_.radius;
    }
    pickSphere_->setRadius( radius );
}

void SurfacePointWidget::preDraw_()
{
    setPointRadius_();
}

void SurfacePointWidget::updateCurrentPosition( const PointOnObject& pos )
{
    currentPos_ = pointOnObjectToPickedPoint( baseObject_.get(), pos );
    updatePositionAndRadius_();
}

void SurfacePointWidget::updateCurrentPosition( const PickedPoint& pos )
{
    currentPos_ = pos;
    updatePositionAndRadius_();
}

bool SurfacePointWidget::isPickIntoBackFace( const std::shared_ptr<MR::VisualObject>& obj, const MR::PointOnObject& pick, const Vector3f& cameraEye )
{
    const auto& xf = obj->worldXf();

    if ( auto objMesh = std::dynamic_pointer_cast< const ObjectMeshHolder >( obj ) )
    {
        const auto& n = objMesh->mesh()->dirDblArea( pick.face );
        if ( dot( xf.A * n, cameraEye ) < 0 )
            return true;
        else
            return false;
    }


    if ( auto objPoints = std::dynamic_pointer_cast< const ObjectPointsHolder >( obj ) )
    {
        if ( objPoints->pointCloud()->normals.size() > static_cast< int > ( pick.vert ) )
        {
            const auto& n = objPoints->pointCloud()->normals[pick.vert];
            auto dt = dot( xf.A * n, cameraEye );
            if ( dt < 0 )
                return true;
            else
                return false;
        }
    }

    return false;
}

}
