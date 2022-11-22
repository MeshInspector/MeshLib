#include "MRTouchesController.h"
#include "MRViewer.h"
#include "MRPch/MRSpdlog.h"
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
        viewer->eventsQueue.emplace( [x,y,viewer] ()
        {
            viewer->mouseMove( x, y ); // to setup position in MouseController
            viewer->draw();
            viewer->mouseDown( MouseButton::Left, 0 );
        } );
        return true;
    }
    if ( mouseMode_ )
    {
        mouseMode_ = false;
        viewer->eventsQueue.emplace( [viewer] ()
        {
            viewer->mouseUp( MouseButton::Left, 0 );
        } );
    }
    return true;
}

bool TouchesController::onTouchMove_( int id, int x, int y )
{
    if ( !multiInfo_.update( { id, Vector2f( float(x), float(y) ) } ) )
        return true;
    auto* viewer = &getViewerInstance();
    if ( mouseMode_ )
    {
        auto eventCall = [x, y, viewer] ()
        {
            viewer->mouseMove( x, y );
            viewer->draw();
        };
        viewer->eventsQueue.emplace( eventCall, true );
    }
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
        viewer->eventsQueue.emplace( [viewer] ()
        {
            viewer->mouseUp( MouseButton::Left, 0 );
        } );
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