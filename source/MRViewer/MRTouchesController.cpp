#include "MRTouchesController.h"
#include "MRViewer.h"
#include "MRPch/MRSpdlog.h"
#include "MRMesh/MR2to3.h"
#include <chrono>

namespace MR
{

bool TouchesController::onTouchStart_( int id, int x, int y )
{
    if ( !multiInfo_.update( { id, Vector2f( float(x), float(y) ) } ) )
        return true;
    auto* viewer = &getViewerInstance();
    auto numPressed = multiInfo_.getNumPressed();
    auto finger = *multiInfo_.getFingerById( id );
    if ( finger == MultiInfo::Finger::First && numPressed == 1 )
    {
        mouseMode_ = true;
        viewer->eventQueue.emplace( { "First touch imitates left mouse down", [x,y,viewer] ()
        {
            viewer->mouseMove( x, y ); // to setup position in MouseController
            viewer->draw();
            viewer->mouseDown( MouseButton::Left, 0 );
        } } );
        return true;
    }
    if ( mouseMode_ )
    {
        mouseMode_ = false;
        viewer->eventQueue.emplace( { "First touch imitates left mouse up", [viewer] ()
        {
            viewer->mouseUp( MouseButton::Left, 0 );
        } } );
    }
    return true;
}

bool TouchesController::onTouchMove_( int id, int x, int y )
{
    if ( !multiInfo_.update( { id, Vector2f( float(x), float(y) ) } ) )
        return true;
    auto* viewer = &getViewerInstance();
    Viewer::EventQueue::EventCallback eventCall;
    if ( mouseMode_ )
    {
        eventCall = [x, y, viewer] ()
        {
            viewer->mouseMove( x, y );
            viewer->draw();
        };
    }
    else if ( multiInfo_.getNumPressed() == 2 && ( touchModeMask_ & ModeBit::Any ) )
    {
        eventCall = [info = multiInfo_, prevInfoPtr = &multiPrevInfo_, viewer, modeMask = touchModeMask_]() mutable
        {
            auto& prevInfoRef = *prevInfoPtr;
            if ( !prevInfoRef.getIdByFinger( MultiInfo::Finger::First ) || 
                 !prevInfoRef.getIdByFinger( MultiInfo::Finger::Second ) )
                 prevInfoRef = info;
            auto oldPos0 = *prevInfoRef.getPosition( MultiInfo::Finger::First );
            auto oldPos1 = *prevInfoRef.getPosition( MultiInfo::Finger::Second );

            auto newPos0 = *info.getPosition( MultiInfo::Finger::First );
            auto newPos1 = *info.getPosition( MultiInfo::Finger::Second );

            auto& vp = viewer->viewport();
            Vector3f sceneCenter;
            if ( vp.getSceneBox().valid() )
                sceneCenter = vp.getSceneBox().center();
            auto sceneCenterVpZ = vp.projectToViewportSpace( sceneCenter ).z;

            auto oldVpPos0 = viewer->screenToViewport( Vector3f( oldPos0.x, oldPos0.y, sceneCenterVpZ ), vp.id );
            auto oldVpPos1 = viewer->screenToViewport( Vector3f( oldPos1.x, oldPos1.y, sceneCenterVpZ ), vp.id );
            auto newVpPos0 = viewer->screenToViewport( Vector3f( newPos0.x, newPos0.y, sceneCenterVpZ ), vp.id );
            auto newVpPos1 = viewer->screenToViewport( Vector3f( newPos1.x, newPos1.y, sceneCenterVpZ ), vp.id );

            auto oldWorldPos0 = vp.unprojectFromViewportSpace( oldVpPos0 );
            auto oldWorldPos1 = vp.unprojectFromViewportSpace( oldVpPos1 );
            auto newWorldPos0 = vp.unprojectFromViewportSpace( newVpPos0 );
            auto newWorldPos1 = vp.unprojectFromViewportSpace( newVpPos1 );

            AffineXf3f aggregateXf;
            auto oldWorldCenter = ( oldWorldPos0 + oldWorldPos1 ) * 0.5f;
            auto newWorldCenter = ( newWorldPos0 + newWorldPos1 ) * 0.5f;
            // TRANSLATION
            if ( modeMask & ModeBit::Translate )
            {
                aggregateXf = AffineXf3f::translation( newWorldCenter - oldWorldCenter );
            }

            // ROTATION
            if ( modeMask & ModeBit::Rotate )
            {
                auto a = ( oldWorldPos1 - oldWorldPos0 ).normalized();
                auto b = ( newWorldPos1 - newWorldPos0 ).normalized();
                // apply rotation first
                aggregateXf = aggregateXf * AffineXf3f::xfAround( Matrix3f::rotation( a, b ), oldWorldCenter  );
            }

            // ZOOM
            if ( modeMask & ModeBit::Zoom )
            {
                auto cameraPoint = vp.getCameraPoint();
                auto vpCenter = vp.unprojectFromClipSpace( Vector3f( 0.0f, 0.0f, sceneCenterVpZ * 2.0f - 1.0f ) );
                auto mult = angle( oldWorldPos0 - cameraPoint, oldWorldPos1 - cameraPoint ) /
                            angle( newWorldPos0 - cameraPoint, newWorldPos1 - cameraPoint );
                constexpr float minAngle = 0.001f;
                constexpr float maxAngle = 179.99f;
                vp.setCameraViewAngle( std::clamp( vp.getParameters().cameraViewAngle * mult, minAngle, maxAngle ) );
                aggregateXf = AffineXf3f::translation( ( newWorldCenter - vpCenter ) * ( mult - 1.0f ) ) * aggregateXf;
            }

            vp.transformView( aggregateXf );
            prevInfoRef = info;
        };
    }
    else
        return true;
    viewer->eventQueue.emplace( { "Two touches move", eventCall }, true );
    return true;
}

bool TouchesController::onTouchEnd_( int id, int x, int y )
{
    if ( !multiInfo_.update( { id, Vector2f( float(x), float(y) ) }, true ) )
        return true;
    auto* viewer = &getViewerInstance();
    if ( mouseMode_ )
    {
        mouseMode_= false;
        viewer->eventQueue.emplace( { "First touch imitates left mouse up", [viewer] ()
        {
            viewer->mouseUp( MouseButton::Left, 0 );
        } } );
    }
    else
    {
        viewer->eventQueue.emplace( { "Touch up", [info = multiInfo_, prevInfoPtr = &multiPrevInfo_] ()
        {
            *prevInfoPtr = info;
        } });
    }
    return true;
}

bool TouchesController::MultiInfo::update( TouchesController::Info info, bool remove )
{
    Info* thisInfoPtr{nullptr};
    if ( info.id == info_[0].id )
        thisInfoPtr = &info_[0];
    else if ( info.id == info_[1].id )
        thisInfoPtr = &info_[1];
    
    if ( remove )
    {
        if ( !thisInfoPtr )
            return false;
        thisInfoPtr->id = -1;
        return true;
    }
    if ( !thisInfoPtr && info_[1].id == -1 )
    {
        if ( info_[0].id == -1 )
            thisInfoPtr = &info_[0];
        else
            thisInfoPtr = &info_[1];
    }
    if ( !thisInfoPtr )
        return false;
    *thisInfoPtr = std::move( info );
    return true;
}

std::optional<Vector2f> TouchesController::MultiInfo::getPosition( Finger fing ) const
{
    const auto& infoRef = info_[int(fing)];
    if ( infoRef.id != -1 )
        return infoRef.position;
    return {};
}

std::optional<Vector2f> TouchesController::MultiInfo::getPosition( int id ) const
{
    if ( info_[0].id == id )
        return info_[0].position;
    if ( info_[1].id == id )
        return info_[1].position;
    return {};
}

int TouchesController::MultiInfo::getNumPressed() const
{
    int counter = 0;
    if ( info_[0].id != -1 )
        ++counter;
    if ( info_[1].id != -1 )
        ++counter;
    return counter;
}

std::optional<TouchesController::MultiInfo::Finger> TouchesController::MultiInfo::getFingerById( int id ) const
{
    if ( info_[0].id == id )
        return Finger::First;
    if ( info_[1].id == id )
        return Finger::Second;
    return {};
}

std::optional<int> TouchesController::MultiInfo::getIdByFinger( Finger fing ) const
{
    auto id = info_[int(fing)].id;
    if ( id == -1 )
        return {};
    return id;
}

}