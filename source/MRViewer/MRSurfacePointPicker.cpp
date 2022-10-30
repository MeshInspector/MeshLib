#include "MRSurfacePointPicker.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRSphereObject.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRRingIterator.h"

namespace MR
{

SurfacePointWidget::~SurfacePointWidget()
{
    reset();
}

const MeshTriPoint& SurfacePointWidget::create( const std::shared_ptr<ObjectMesh>& surface, const MeshTriPoint& startPos )
{
    reset();
    if ( !surface || !surface->mesh() )
        return startPos;
    baseSurface_ = surface;
    
    pickSphere_ = std::make_shared<SphereObject>();
    pickSphere_->setName( "Pick Sphere" );
    pickSphere_->setAncillary( true );
    pickSphere_->setFrontColor( params_.baseColor, false );
    baseSurface_->addChild( pickSphere_ );
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

    baseSurface_.reset();

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
    }
}

bool SurfacePointWidget::onMouseDown_( Viewer::MouseButton button, int mod )
{
    if ( button != MouseButton::Left || mod != 0 || !isHovered_ )
        return false;

    pickSphere_->setPickable( false );
    isOnMove_ = true;
    pickSphere_->setFrontColor( params_.activeColor, false );
    if ( startMove_ )
        startMove_( currentPos_ );
    return true;
}

bool SurfacePointWidget::onMouseUp_( Viewer::MouseButton button, int )
{
    if ( button != MouseButton::Left || !isOnMove_)
        return false;
    isOnMove_ = false;
    pickSphere_->setPickable( true );
    pickSphere_->setFrontColor( params_.baseColor, false );
    if ( endMove_ )
        endMove_( currentPos_ );
    return true;
}

bool SurfacePointWidget::onMouseMove_( int, int )
{
    if ( isOnMove_ )
    {
        auto [obj, pick] = getViewerInstance().viewport().pick_render_object();
        if ( obj != baseSurface_ )
            return false;
        
        currentPos_ = baseSurface_->mesh()->toTriPoint( pick );
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

void SurfacePointWidget::updatePositionAndRadius_()
{
    assert( pickSphere_ );
    assert( baseSurface_ );
    assert( baseSurface_->mesh() );
    const auto& mesh = *baseSurface_->mesh();
    FaceId fId = mesh.topology.left( currentPos_.e );
    switch ( params_.positionType )
    {
        case PositionType::FaceCenters:
        {
            currentPos_ = mesh.toTriPoint( fId, mesh.triCenter( fId ) );
            break;
        }
        case PositionType::Edges:
        {
            if ( !currentPos_.onEdge( mesh.topology ) )
            {
                auto closestEdge = EdgeId( mesh.getClosestEdge( PointOnFace{ fId,mesh.triPoint( currentPos_ ) } ) );
                if ( mesh.topology.left( closestEdge ) != fId )
                    closestEdge = closestEdge.sym();
                if ( currentPos_.e == closestEdge )
                    currentPos_.bary.b = 0.0f;
                else if ( currentPos_.e == mesh.topology.next( closestEdge ).sym() )
                {
                    currentPos_.e = closestEdge;
                    currentPos_.bary.a = currentPos_.bary.b;
                    currentPos_.bary.b = 0.0f;
                }
                else
                {
                    currentPos_.e = closestEdge;
                    currentPos_.bary.a = 1.0f - currentPos_.bary.b;
                    currentPos_.bary.b = 0.0f;
                }
            }
            break;
        }
        case PositionType::EdgeCeneters:
        {
            auto closestEdge = EdgeId( mesh.getClosestEdge( PointOnFace{ fId,mesh.triPoint( currentPos_ ) } ) );
            if ( mesh.topology.left( closestEdge ) != fId )
                closestEdge = closestEdge.sym();
            currentPos_.e = closestEdge;
            currentPos_.bary.a = 0.5f;
            currentPos_.bary.b = 0.0f;
            break;
        }
        case PositionType::Verts:
        {
            if ( !currentPos_.inVertex() )
            {
                auto closestVert = mesh.getClosestVertex( PointOnFace{ fId,mesh.triPoint( currentPos_ ) } );
                for ( auto e : orgRing( mesh.topology, closestVert ) )
                {
                    if ( mesh.topology.left( e ) == fId )
                    {
                        currentPos_.e = e;
                        currentPos_.bary.a = 0.0f;
                        currentPos_.bary.b = 0.0f;
                        break;
                    }
                }
            }
            break;
        }
        default:
            break;
    }
    float radius = params_.radius <= 0.0f ? mesh.getBoundingBox().diagonal() * 5e-3f : params_.radius;
    pickSphere_->setCenter( mesh.triPoint( currentPos_ ) );
    pickSphere_->setRadius( radius );
}

}