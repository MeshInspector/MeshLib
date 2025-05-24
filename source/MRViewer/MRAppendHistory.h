#pragma once

#include "MRHistoryStore.h"
#include <MRMesh/MRHistoryAction.h>
#include <string>
#include <memory>

namespace MR
{

/// Appends given history action to viewer's global history store
inline void AppendHistory( std::shared_ptr<HistoryAction> action )
{
    if ( const auto & s = HistoryStore::getViewerInstance() )
        s->appendAction( std::move( action ) );
}

/// Constructs history action from given arguments, than appends it to viewer's global history store
template<class HistoryActionType, typename... Args>
void AppendHistory( Args&&... args )
{
    static_assert( std::is_base_of_v<HistoryAction, HistoryActionType> );
    // even if there is no HistoryStore::getViewerInstance(), we still need to make new action,
    // because some actions make modifications visible outside in their constructors
    AppendHistory( std::make_shared<HistoryActionType>( std::forward<Args>( args )... ) );
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
        if ( HistoryStore::getViewerInstance() )
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

    ~Historian()
    {
        if ( action_ )
            AppendHistory( std::move( action_ ) );
        if ( !canceled_ )
            HistoryActionType::setObjectDirty( obj_ );
    }
private:
    std::shared_ptr<Obj> obj_;
    std::shared_ptr<HistoryActionType> action_;
    bool canceled_{ false };
};

/// Remove actions from global history (if it is enabled) that match the condition
/// @param deepFiltering - filter actions into combined actions
MRVIEWER_API void FilterHistoryByCondition( HistoryStackFilter filteringCondition, bool deepFiltering = true );

/// The purpose of this class is to combine all actions appended to global history store in one big action to undo/redo them all at once.
class ScopeHistory
{
public:
    /// creates new CombinedHistoryAction, and setups global history store to append all new actions there during this object lifetime
    MRVIEWER_API ScopeHistory( const std::string& name );

    /// created before CombinedHistoryAction if not empty is appended (with all sub-actions) in the global history store
    MRVIEWER_API ~ScopeHistory();

    /// returns the action being populated
    const std::shared_ptr<CombinedHistoryAction>& combinedAction() const { return combinedAction_; }

private:
    std::shared_ptr<HistoryStore> store_;
    std::shared_ptr<CombinedHistoryAction> combinedAction_;
    HistoryActionsVector* parentScopePtr_{ nullptr };
};

#define SCOPED_HISTORY(name) MR::ScopeHistory __startScopedHistoryMode(name)

}
