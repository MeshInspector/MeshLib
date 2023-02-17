#pragma once

#include "MRViewer.h"
#include <MRMesh/MRHistoryAction.h>
#include "MRBlockingStateSession.h"
#include <string>
#include <memory>

namespace MR
{

// This function appends history action to viewers global history store
template<class HistoryActionType, typename... Args>
void AppendHistory( Args&&... args )
{
    static_assert( std::is_base_of_v<HistoryAction, HistoryActionType> );
    if ( getViewerInstance().isGlobalHistoryEnabled() )
        getViewerInstance().appendHistoryAction( std::make_shared<HistoryActionType>( std::forward<Args>( args )... ) );
}

template<class HistoryActionType>
void AppendHistory( std::shared_ptr<HistoryActionType> action )
{
    static_assert( std::is_base_of_v<HistoryAction, HistoryActionType> );
    getViewerInstance().appendHistoryAction( action );
}

template<class HistoryActionType, typename... Args>
void AppendBlockingHistory( Args&&... args )
{
    static_assert( std::is_base_of_v<HistoryAction, HistoryActionType> );
    if ( getViewerInstance().isGlobalHistoryEnabled() )
    {
        auto action = std::make_shared<HistoryActionType>( std::forward<Args>( args )... );
        action->sessionId = BlockingStateSession::getCurrentSessionId();
        getViewerInstance().appendHistoryAction( action );
    }
}

template<class HistoryActionType>
void AppendBlockingHistory( std::shared_ptr<HistoryActionType> action )
{
    static_assert( std::is_base_of_v<HistoryAction, HistoryActionType> );
    action->sessionId = BlockingStateSession::getCurrentSessionId();
    getViewerInstance().appendHistoryAction( action );
}

// if undo history is enabled, creates given action in the constructor;
// and always mark the object as dirty in the destructor
template<class HistoryActionType>
class Historian
{
public:
    static_assert( std::is_base_of_v<HistoryAction, HistoryActionType> );
    using Obj = typename HistoryActionType::Obj;

    template<typename... Args>
    Historian( std::string name, std::shared_ptr<Obj> obj, Args&&... args ) : obj_( std::move( obj ) )
    {
        if ( getViewerInstance().isGlobalHistoryEnabled() )
            action_ = std::make_shared<HistoryActionType>( std::move( name ), obj_, std::forward<Args>( args )... );
    }

    void cancelAction()
    {
        if ( action_ )
        {
            action_->action( HistoryAction::Type::Undo );
            action_.reset();
        }
        canceled_ = true;
    }

    virtual ~Historian()
    {
        if ( action_ )
            getViewerInstance().appendHistoryAction( action_ );
        if ( !canceled_ )
            HistoryActionType::setObjectDirty( obj_ );
    }
protected:
    std::shared_ptr<Obj> obj_;
    std::shared_ptr<HistoryActionType> action_;
    bool canceled_{ false };
};

template<class HistoryActionType>
class BlockingHistorian : public Historian<HistoryActionType>
{
public:
    using Parent = Historian<HistoryActionType>;
    using Obj = typename HistoryActionType::Obj;

    template<typename... Args>
    BlockingHistorian( std::string name, std::shared_ptr<Obj> obj, Args&&... args ) :
        Parent( std::move( name ), std::move( obj ), std::forward<Args>( args )... )
    {
        if ( this->action_ )
            this->action_->sessionId = BlockingStateSession::getCurrentSessionId();
    }
};

/// Remove actions from global history (if it is enabled) that match the condition
/// @param deepFiltering - filter actions into combined actions
MRVIEWER_API void FilterHistoryByCondition( HistoryStackFilter filteringCondition, bool deepFiltering = true );

// This class store history actions that are appended to global history stack all together as CombinedHistoryAction in destructor (if scoped stack is not empty)
class MRVIEWER_CLASS ScopeHistory
{
public:
    MRVIEWER_API ScopeHistory( const std::string& name, bool blocking );
    MRVIEWER_API ~ScopeHistory();

private:
    std::string name_;
    std::shared_ptr<HistoryStore> store_;
    bool blocking_{ false };
    bool thisScopeStartedScopedMode_ = false;
};

#define SCOPED_HISTORY(name) MR::ScopeHistory __startScopedHistoryMode(name,false)

#define SCOPED_BLOCKING_HISTORY(name) MR::ScopeHistory __startScopedBlockingHistoryMode(name,true)

}
