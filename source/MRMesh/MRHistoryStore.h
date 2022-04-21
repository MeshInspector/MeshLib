#pragma once
#include "MRMeshFwd.h"
#include "MRHistoryAction.h"
#include <boost/signals2/signal.hpp>
#include <memory>

namespace MR
{

// This class stores history stack for undo/redo
class HistoryStore
{
public:

    HistoryStore() = default;
    MRMESH_API virtual ~HistoryStore();
    // Adds action in history stack (clears available redo actions)
    // adds actions to scope block if scope mode is active (do not affect main stack)
    MRMESH_API virtual void appendAction( const std::shared_ptr<HistoryAction>& action );

    // Start using scope block for storing actions, or stop and clear it
    MRMESH_API virtual void startScope( bool on );
    // Returns true if scoped mode is active now, false otherwise
    bool isInScopeMode() const { return scoped_; }
    // Returns actions made in scope
    const std::vector<std::shared_ptr<HistoryAction>>& getScopeBlock() const { return scopedBlock_; }

    // Returns true if the current scene state does not match the saved state
    bool isSceneModified() const { return firstRedoIndex_ != savedSceneIndex_; }
    // Consider the current scene state as saved
    void setSavedState() { savedSceneIndex_ = firstRedoIndex_; }

    // Clears this HistoryStore
    MRMESH_API void clear();

    // Set memory limit for this store, if history stack exceed it - old actions are removed 
    void setMemoryLimit( size_t limit ) { storageLimit_ = limit; }
    // Returns current memory limit for this store (by default uint64 max)
    size_t getMemoryLimit() const { return storageLimit_; }

    // remove some actions according to condition
    MRMESH_API void filterStack( HistoryStackFilter filteringCondition, bool deepFiltering = true );

    MRMESH_API virtual bool undo();
    MRMESH_API virtual bool redo();

    // Returns names of last N undo actions or first N redo actions
    MRMESH_API std::vector<std::string> getNActions( unsigned n, HistoryAction::Type type )const;

    // Signal is called after this store changed
    enum class ChangeType
    {
        AppendAction, // additions in scoped block does not provide signal
        Undo,
        Redo,
        Clear,
        Filter
    };
    using HistoryStoreChangedSignal = boost::signals2::signal<void( const HistoryStore& store, ChangeType )>;
    HistoryStoreChangedSignal changedSignal;

private:
    bool scoped_{ false };
    // buffer for merging actions
    HistoryActionsVector scopedBlock_;
    // main history stack
    HistoryActionsVector stack_;
    // this index points to first redo action or equals to size of stack if no redo is available
    size_t firstRedoIndex_{ 0 };
    // this index points to the position in stack_ corresponding to saved scene state;
    // if firstRedoIndex_ == savedSceneIndex_ then the scene is considered as not modified
    size_t savedSceneIndex_{ 0 };
    // memory limit (bytes) to this HistoryStore if stack_ exceed it, old actions are removed
    size_t storageLimit_{ size_t( ~0 ) };
};

/**
 * Remove actions from history actions vector that match the condition
 * @param firstRedoIndex - set redo index for calculate how many actions removed before it
 * @param deepFiltering - filter actions into combined actions
 * @return pair (anything removed, how many removed before firstRedoIndex)
 */
std::pair<bool, int> filterHistoryActionsVector( HistoryActionsVector& historyVector,
    HistoryStackFilter filteringCondition, size_t firstRedoIndex = 0, bool deepFiltering = true );

}
