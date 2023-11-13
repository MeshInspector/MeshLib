#include "MRSurfaceManipulationWidget.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRViewerInstance.h"
#include "MRAppendHistory.h"
#include "MRViewer/MRGladGlfw.h"

#include "MRMouse.h"
#include "MRMesh/MREdgePaths.h"
#include "MRMesh/MRPositionVertsSmoothly.h"
#include "MRMesh/MRSurfaceDistance.h"
#include "MRMesh/MRExpandShrink.h"
#include "MRMesh/MREnumNeighbours.h"
#include "MRMesh/MRTimer.h"
#include <MRMesh/MRMeshRelax.h>
#include <MRMesh/MRBitSetParallelFor.h>


namespace MR
{
//const float k = r < 1-a ? std::sqrt( sqr( 1 - a ) - sqr( r ) ) + ( 1 - a ) : -std::sqrt( sqr( a ) - sqr( r - 1 ) ) + a; // alternative version F_point_shift(r,i) (i == a)

void SurfaceManipulationWidget::init( const std::shared_ptr<ObjectMesh>& objectMesh )
{
    obj_ = objectMesh;
    diagonal_ = obj_->getBoundingBox().diagonal();
    minRadius_ = obj_->avgEdgeLen() * 2.f;
    settings_.radius = diagonal_ * 0.025f;

    size_t numV = obj_->mesh()->topology.lastValidVert() + 1;
    region_ = VertBitSet( numV, false );
    regionExpanded_ = VertBitSet( numV, false );
    distances_ = VertScalars( numV, 0.f );

    obj_->setAncillaryTexture( { { { Color { 255, 64, 64, 255 }, Color { 0, 0, 0, 0 } }, Vector2i { 1, 2 } } } );
    uvs_ = VertUVCoords( numV, { 0, 1 } );
    obj_->setAncillaryUVCoords( uvs_ );

    workMode_ = WorkMode::Add;

    connect( &getViewerInstance() );
}

void SurfaceManipulationWidget::reset()
{
    obj_->clearAncillaryTexture();
    obj_.reset();

    region_ = {};
    regionExpanded_ = {};
    distances_ = {};

    uvs_ = {};

    changeMeshAction_.reset();

    disconnect();
}

void SurfaceManipulationWidget::setSettings( const Settings& settings )
{
    settings_ = settings;
    settings_.radius = std::max( settings_.radius, minRadius_ );
    settings_.force = std::clamp( settings_.force, 1.f, 100.f );
    settings_.saturation = std::clamp( settings_.saturation, 1.f, 100.f );
}

bool SurfaceManipulationWidget::onMouseDown_( Viewer::MouseButton button, int /*modifier*/ )
{
    if ( button != MouseButton::Left )
        return false;

    auto [obj, pick] = getViewerInstance().viewport().pick_render_object();
    if ( !obj || obj != obj_ )
        return false;

    changeMeshAction_ = std::make_shared<ChangeMeshAction>( "Change mesh surface", obj_ );
    mousePressed_ = true;
    timePoint_ = std::chrono::high_resolution_clock::now();

    return true;
}

bool SurfaceManipulationWidget::onMouseUp_( Viewer::MouseButton button, int /*modifier*/ )
{
    if ( button != MouseButton::Left )
        return false;

    mousePressed_ = false;

    AppendHistory( changeMeshAction_ );
    changeMeshAction_.reset();

    return true;
}

bool SurfaceManipulationWidget::onMouseMove_( int mouse_x, int mouse_y )
{
    updateRegion_( Vector2f{ float( mouse_x ), float( mouse_y ) } );

    return true;
}

bool SurfaceManipulationWidget::onKeyDown_( int /*key*/, int modifier )
{
    bool res = false;
    WorkMode newWorkMode = WorkMode::Add;
    if ( modifier & GLFW_MOD_SHIFT )
    {
        newWorkMode = WorkMode::Relax;
        res = true;
    }
    else if ( modifier & GLFW_MOD_CONTROL )
    {
        newWorkMode = WorkMode::Remove;
        res = true;
    }
    if ( newWorkMode != WorkMode::Relax && workMode_ == WorkMode::Relax )
        timePoint_ = std::chrono::high_resolution_clock::now();
    workMode_ = newWorkMode;

    return res;
}

bool SurfaceManipulationWidget::onKeyUp_( int /*key*/, int modifier )
{
    bool res = false;
    WorkMode newWorkMode = WorkMode::Add;
    if ( modifier & GLFW_MOD_SHIFT )
    {
        newWorkMode = WorkMode::Relax;
        res = true;
    }
    else if ( modifier & GLFW_MOD_CONTROL )
    {
        newWorkMode = WorkMode::Remove;
        res = true;
    }
    if ( newWorkMode != WorkMode::Relax && workMode_ == WorkMode::Relax )
        timePoint_ = std::chrono::high_resolution_clock::now();
    workMode_ = newWorkMode;

    return res;
}

void SurfaceManipulationWidget::preDraw_()
{
    if ( mousePressed_ )
        changeSurface_();
    
    updateRegion_( mousePos_ );
}

void SurfaceManipulationWidget::changeSurface_()
{
    if ( !region_.any() )
        return;

    MR_TIMER;

    if ( workMode_ == WorkMode::Relax )
    {
        MeshRelaxParams params;
        params.region = &region_;
        params.force = settings_.force / 200.f; // [1-100] -> (0.0, 0.5]
        relax( *obj_->varMesh(), params );
        obj_->setDirtyFlags( DIRTY_POSITION );
        return;
    }

    Vector3f normal;
    const auto& mesh = *obj_->mesh();
    for ( auto v : region_ )
        normal += mesh.normal( v );
    normal = normal.normalized();

    auto& points = obj_->varMesh()->points;

    auto newTimePoint = std::chrono::high_resolution_clock::now();
    auto timeForce = std::chrono::duration_cast< std::chrono::milliseconds >( newTimePoint - timePoint_ ).count() * 0.06f;
    timePoint_ = newTimePoint;

    const float maxShift = ( settings_.force / 5.f + 10.f ) / 1000.f * diagonal_ / 10.f * timeForce;
    const float intensity = settings_.saturation / 100.f * 0.15f + 0.4f;
    const float a1 = -1.f * ( 1 - intensity ) / intensity / intensity;
    const float a2 = intensity / ( 1 - intensity ) / ( 1 - intensity );
    const float direction = workMode_ == WorkMode::Remove ? -1.f : 1.f;
    BitSetParallelFor( region_, [&] ( VertId v )
    {
        const float r = std::clamp( distances_[v] / settings_.radius, 0.f, 1.f );
        const float k = r < intensity ? a1 * r * r + 1 : a2 * ( r - 1 ) * ( r - 1 ); // I(r)
        float pointShift = maxShift * k; // shift = F * I(r)
        points[v] += direction * pointShift * normal;
    } );
    obj_->setDirtyFlags( DIRTY_PRIMITIVES );
}

void SurfaceManipulationWidget::updateUV_( bool set )
{
    BitSetParallelFor( regionExpanded_, [&] ( VertId v )
    {
        uvs_[v] = set ? UVCoord{ 0, std::clamp( distances_[v] / settings_.radius / 2.f, 0.f, 1.f ) } : UVCoord{ 0, 1 };
    } );
}

void SurfaceManipulationWidget::updateRegion_( const Vector2f & mousePos )
{
    if ( mousePos == mousePos_ && !mouseMoved_ )
        return;

    MR_TIMER;

    const auto& viewerRef = getViewerInstance();
    std::vector<ObjAndPick> objAndPick;
    if ( ( mousePos - mousePos_ ).lengthSq() < 25.f )
    {
        mouseMoved_ = false;
        objAndPick.push_back( getViewerInstance().viewport().pick_render_object());
    }
    else
    {
        const Vector2f newMousePos = Vector2f( viewerRef.screenToViewport( Vector3f( mousePos ), viewerRef.getHoveredViewportId() ) );
        const Vector2f oldMousePos = Vector2f( viewerRef.screenToViewport( Vector3f( mousePos_ ), viewerRef.getHoveredViewportId() ) );
        const Vector2f vec = newMousePos - oldMousePos;
        const int count = int( std::ceil( vec.length() / 10.f ) ) + 1;
        const Vector2f step = vec / ( count - 1.f );
        std::vector<Vector2f> points( count );
        for ( int i = 0; i < count; ++i )
            points[i] = oldMousePos + step * float( i );

        objAndPick = getViewerInstance().viewport().multiPickObjects( { obj_.get() }, points );
        mouseMoved_ = true;
    }
    mousePos_ = mousePos;

    const auto& mesh = *obj_->mesh();
    std::vector<MeshTriPoint> triPoints;
    VertBitSet newVerts( region_.size() );
    newVerts.reset(); // realy need?
    triPoints.reserve( objAndPick.size() );
    for ( int i = 0; i < objAndPick.size(); ++i )
    {
        if ( objAndPick[i].first == obj_ )
        {
            const auto& pick = objAndPick[i].second;
            VertId v[3];
            mesh.topology.getTriVerts( pick.face, v );
            for ( int j = 0; j < 3; ++j )
                newVerts.set( v[j] );
            triPoints.push_back( mesh.toTriPoint( pick.face, pick.point ) );
        }
    }

    updateUV_( false );

    if ( triPoints.empty() )
    {
        obj_->setAncillaryUVCoords( uvs_ );
        return;
    }

    region_.reset();
    regionExpanded_ = newVerts;

    dilateRegion( mesh, regionExpanded_, settings_.radius * 2 );

    if ( triPoints.size() == 1 )
        distances_ = computeSpaceDistances( mesh, { mesh.topology.left( triPoints[0].e ), mesh.triPoint( triPoints[0] ) }, settings_.radius * 1.5f );
    else
        distances_ = computeSurfaceDistances( mesh, triPoints, settings_.radius * 1.5f, &regionExpanded_ );

    for ( auto v : regionExpanded_ )
        region_.set( v, distances_[v] <= settings_.radius );

    updateUV_( true );
    obj_->setAncillaryUVCoords( uvs_ );
}

}
