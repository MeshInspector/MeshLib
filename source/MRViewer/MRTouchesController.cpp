#include "MRTouchesController.h"
#include "MRViewer.h"
#include "MRPch/MRSpdlog.h"
#include <chrono>

namespace MR
{

bool TouchesController::onTouchStart_( int id, int x, int y )
{
    if ( id > 1 )
        return true;
    setPos_( id, x, y, true );
    auto& viewer = getViewerInstance();
    if ( id == 0 )
    {
        viewer.mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Move, [x,y] ()
        {
            getViewerInstance().mouseMove( x, y ); // to setup position in MouseController
            getViewerInstance().draw();
        } );
        viewer.mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Down, [] ()
        {
            getViewerInstance().mouseDown( MouseButton::Left, 0 ); // to setup position in MouseController
        } );
        mode_ = MouseButton::Left;
    }
    else if ( id == 1 )
    {
        getViewerInstance().mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Up, [] ()
        {
            getViewerInstance().mouseUp( MouseButton::Left, 0 );
        } );
        getViewerInstance().mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Move, [x,y] ()
        {
            getViewerInstance().mouseMove( x, y ); // to setup position in MouseController
            getViewerInstance().draw();
        } );
        viewer.mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Down, [] ()
        {
            getViewerInstance().mouseDown( MouseButton::Right, 0 ); // to setup position in MouseController
        } );
        mode_ = MouseButton::Right;
        startDist_ = ( Vector2f( *positions_[0] ) - Vector2f( *positions_[1] ) ).length();
        secondTouchStartTime_ = std::chrono::milliseconds( std::chrono::system_clock::now().time_since_epoch().count() ).count();
        blockZoom_ = false;
    }
    return true;
}

bool TouchesController::onTouchMove_( int id, int x, int y )
{
    if ( id > 1 )
        return true;
    auto oldPos = *positions_[id];
    setPos_( id, x, y, true );
    if ( mode_ == MouseButton::Right && !blockZoom_ )
    {
        size_t nowTime = std::chrono::milliseconds( std::chrono::system_clock::now().time_since_epoch().count() ).count();
        if ( nowTime - secondTouchStartTime_ > 7000 )
        {
            auto newDist = ( Vector2f( *positions_[0] ) - Vector2f( *positions_[1] ) ).length();
            auto dist = newDist - startDist_;
            if ( std::abs( dist ) > 30 )
                mode_ = MouseButton::Middle;
            else
                blockZoom_ = true;
        }
    }

    if ( mode_ == MouseButton::Middle )
    {
        auto oldDsit = ( Vector2f( oldPos ) - Vector2f( *positions_[id^1] ) ).length();
        auto newDist = ( Vector2f( *positions_[0] ) - Vector2f( *positions_[1] ) ).length();
        auto dist = newDist - oldDsit;
        getViewerInstance().mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Up, [] ()
        {
            getViewerInstance().mouseUp( MouseButton::Right, 0 );
        } );
        getViewerInstance().mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Move, [x,y] ()
        {
            getViewerInstance().mouseMove( x, y ); // to setup position in MouseController
        } );
        getViewerInstance().mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Move, [dist] ()
        {
            getViewerInstance().mouseScroll( dist / 80.0f );
        } );
        return true;
    }

    if ( mode_ == MouseButton::Right && id == 0 )
        return true;
    auto eventCall = [x, y] ()
    {
        getViewerInstance().mouseMove( int( x ), int( y ) );
        getViewerInstance().draw();
    };
    if ( getViewerInstance().mouseEventQueue.empty() ||
         getViewerInstance().mouseEventQueue.back().type != MR::Viewer::MouseQueueEvent::Type::Move )
    {
        getViewerInstance().mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Move, std::move( eventCall ) );
    }
    else
    {
        // if last event in this frame was move - replace it with newer one
        getViewerInstance().mouseEventQueue.back().callEvent = std::move( eventCall );
    }
    return true;
}

bool TouchesController::onTouchEnd_( int id, int x, int y )
{
    mode_ = MouseButton::Count;
    if ( id > 1 )
        return true;
    setPos_( id, x, y, false );
    getViewerInstance().mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Up, [id] ()
    {
        getViewerInstance().mouseUp( MouseButton( id ), 0 );
    } );
    return true;
}

void TouchesController::setPos_( int id, int x, int y, bool on )
{
    if ( !on )
    {
        if ( positions_.size() > id )
            positions_[id] = {};
        return;
    }
    if ( positions_.size() <= id )
        positions_.resize( id + 1 );
    positions_[id] = Vector2i( x, y );
}

}