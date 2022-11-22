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
        viewer->eventQueue.emplace( false, [x,y,viewer] ()
        {
            viewer->mouseMove( x, y ); // to setup position in MouseController
            viewer->draw();
            viewer->mouseDown( MouseButton::Left, 0 ); // to setup position in MouseController
        } );
        mode_ = MouseButton::Left;
    }
    else if ( finger == MultiInfo::Finger::Second )
    {
        assert( numPressed == 2 );
        viewer->eventQueue.emplace( false, [x,y,viewer] ()
        {
            viewer->mouseUp( MouseButton::Left, 0 );
            viewer->mouseMove( x, y ); // to setup position in MouseController
            viewer->draw();
            viewer->mouseDown( MouseButton::Right, 0 ); // to setup position in MouseController
        } );
        mode_ = MouseButton::Right;
        startDist_ = ( *multiInfo_.getPosition( MultiInfo::Finger::Second ) - 
                       *multiInfo_.getPosition( MultiInfo::Finger::First ) ).length();
        secondTouchStartTime_ = std::chrono::milliseconds( std::chrono::system_clock::now().time_since_epoch().count() ).count();
        blockZoom_ = false;
    }
    return true;
}

bool TouchesController::onTouchMove_( int id, int x, int y )
{
    auto oldPos = multiInfo_.getPosition( id );
    if ( !oldPos )
        return true;
    multiInfo_.update( { id, Vector2f( float(x), float(y) ) } );
    if ( mode_ == MouseButton::Right && !blockZoom_ )
    {
        assert( multiInfo_.getNumPressed() == 2 );
        size_t nowTime = std::chrono::milliseconds( std::chrono::system_clock::now().time_since_epoch().count() ).count();
        if ( nowTime - secondTouchStartTime_ > 7000 )
        {
            auto newDist = ( *multiInfo_.getPosition( MultiInfo::Finger::Second ) - 
                             *multiInfo_.getPosition( MultiInfo::Finger::First ) ).length();
            auto dist = newDist - startDist_;
            if ( std::abs( dist ) > 30 )
                mode_ = MouseButton::Middle;
            else
                blockZoom_ = true;
        }
    }
    auto* viewer = &getViewerInstance();
    if ( mode_ == MouseButton::Middle )
    {
        assert( multiInfo_.getNumPressed() == 2 );
        auto finger = *multiInfo_.getFingerById( id );
        auto oldDsit = ( *oldPos - 
                         *multiInfo_.getPosition( MultiInfo::Finger( int( finger )^1 ) ) ).length();
        auto newDist = ( *multiInfo_.getPosition( MultiInfo::Finger::Second ) - 
                         *multiInfo_.getPosition( MultiInfo::Finger::First ) ).length();
        auto dist = newDist - oldDsit;
        viewer->eventQueue.emplace( false, [x,y,dist,viewer] ()
        {
            viewer->mouseUp( MouseButton::Right, 0 );
            viewer->mouseMove( x, y ); // to setup position in MouseController
            viewer->mouseScroll( dist / 80.0f );
        } );
        return true;
    }
    if ( mode_ == MouseButton::Right && *multiInfo_.getFingerById( id ) == MultiInfo::Finger::First )
        return true;

    auto eventCall = [x, y,viewer] ()
    {
        viewer->mouseMove( x, y );
        viewer->draw();
    };
    if ( viewer->eventQueue.empty() ||
         !viewer->eventQueue.back().skipable )
    {
        viewer->eventQueue.emplace( true, std::move( eventCall ) );
    }
    else
    {
        // if last event in this frame was skipable - replace it with newer one
        viewer->eventQueue.back().callEvent = std::move( eventCall );
    }
    return true;
}

bool TouchesController::onTouchEnd_( int id, int x, int y )
{
    mode_ = MouseButton::Count;
    auto fing = multiInfo_.getFingerById( id );
    if ( !fing )
        return true;
    multiInfo_.update( { id, Vector2f( float(x), float(y) ) }, true );
    auto* viewer = &getViewerInstance();
    viewer->eventQueue.emplace( false, 
        [mb = int( *fing ),viewer] ()
    {
        viewer->mouseUp( MouseButton( mb ), 0 );
    } );
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