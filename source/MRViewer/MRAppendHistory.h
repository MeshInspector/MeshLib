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

/// The main objective of this class is to save object's state in the constructor,
/// then let the caller modify object's data in-place,
/// and finally append the action in the history store and automatically call appropriate setDirty in the destructor.
/// If HistoryStore::getViewerInstance() is missing, it does not allocate memory for undo (the action is not created at all).
/// Always create a named Historian variable and never a nameless temporary such as `Historian<ChangeMeshPointsAction>( "name", obj );`
/// because a temporary is destroyed at the end of the same statement, calling setDirty before any data modification.
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
    MRVIEWER_API ScopeHistory( const std::string& name, const std::function<std::string ()>& dynamicNameFunc = {} );

    /// created before CombinedHistoryAction if not empty is appended (with all sub-actions) in the global history store
    MRVIEWER_API ~ScopeHistory();

    /// returns the action being populated
    const std::shared_ptr<CombinedHistoryAction>& combinedAction() const { return combinedAction_; }

private:
    HistoryStore* store_{ nullptr };
    std::shared_ptr<CombinedHistoryAction> combinedAction_;
    HistoryActionsVector* parentScopePtr_{ nullptr };
};

#define SCOPED_HISTORY( ... ) MR::ScopeHistory __startScopedHistoryMode( __VA_ARGS__ )

}
