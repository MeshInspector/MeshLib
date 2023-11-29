#include "MRViewerEventQueue.h"

namespace MR
{

void ViewerEventQueue::emplace( std::string name, ViewerEventCallback cb, bool skipable )
{
    std::unique_lock lock( mutex_ );
    if ( queue_.empty() || !skipable || !lastSkipable_ )
        queue_.emplace( NamedEvent{ std::move( name ), std::move( cb ) } );
    else
        queue_.back() = { std::move( name ), std::move( cb ) };
    lastSkipable_ = skipable;
}

void ViewerEventQueue::execute()
{
    std::unique_lock lock( mutex_ );
    while ( !queue_.empty() )
    {
        if ( queue_.front().cb )
            queue_.front().cb();
        queue_.pop();
    }
}

bool ViewerEventQueue::empty() const
{
    std::unique_lock lock( mutex_ );
    return queue_.empty();
}

void ViewerEventQueue::popByName( const std::string& name )
{
    std::unique_lock lock( mutex_ );
    while ( !queue_.empty() && queue_.front().name == name )
        queue_.pop();
}

} //namespace MR
