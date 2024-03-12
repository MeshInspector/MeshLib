#pragma once

#include "MRViewerFwd.h"
#include "MRMesh/MRHistoryAction.h"
#include <boost/signals2/signal.hpp>
#include <memory>

namespace MR
{

/// \addtogroup HistoryGroup
/// \{

/// This class stores history stack for undo/redo
class HistoryStore
{
public:
    /// returns the instance (if any) of HistoryStore from the viewer
    MRVIEWER_API static const std::shared_ptr<HistoryStore>& getViewerInstance();

    MRVIEWER_API virtual ~HistoryStore();

    /// Adds action in history stack (clears available redo actions)
    /// adds actions to scope block if scope mode is active (do not affect main stack)
    MRVIEWER_API virtual void appendAction( const std::shared_ptr<HistoryAction>& action );

    /// Returns current scope ptr
    HistoryActionsVector* getScopeBlockPtr() const { return scopedBlock_; }
    /// Sets pointer to current scope block
    void setScopeBlockPtr( HistoryActionsVector* scopedBlock ) { scopedBlock_ = scopedBlock; }

    /// Returns true if the current scene state does not match the saved state
    bool isSceneModified() const { return firstRedoIndex_ != savedSceneIndex_; }
    /// Consider the current scene state as saved
    void setSavedState() { savedSceneIndex_ = firstRedoIndex_; }

    /// Clears this HistoryStore
    MRVIEWER_API void clear();

    /// Set memory limit for this store, if history stack exceed it - old actions are removed 
    void setMemoryLimit( size_t limit ) { storageLimit_ = limit; }
    /// Returns current memory limit for this store (by default uint64 max)
    size_t getMemoryLimit() const { return storageLimit_; }

    /// Returns full history stack
    const HistoryActionsVector& getHistoryStack() const { return stack_; }
    /// Returns index of first redo action in stack
    size_t getStackPointer() const { return firstRedoIndex_; }

    /// remove some actions according to condition
    MRVIEWER_API void filterStack( HistoryStackFilter filteringCondition, bool deepFiltering = true );

    MRVIEWER_API virtual bool undo();
    MRVIEWER_API virtual bool redo();

    /// Returns names of last N undo actions or first N redo actions
    MRVIEWER_API std::vector<std::string> getNActions( unsigned n, HistoryAction::Type type )const;
    /// Returns the name of last undo or redo action (or empty string if there is no such action)
    MRVIEWER_API std::string getLastActionName( HistoryAction::Type type ) const;

    /// Signal is called after this store changed
    enum class ChangeType
    {
        AppendAction, ///< additions in scoped block does not provide signal
        Undo,
        Redo,
        Clear,
        Filter
    };
    using HistoryStoreChangedSignal = boost::signals2::signal<void( const HistoryStore& store, ChangeType )>;
    HistoryStoreChangedSignal changedSignal;

private:
    /// buffer for merging actions, if present, used for storing
    HistoryActionsVector* scopedBlock_{ nullptr };
    /// main history stack
    HistoryActionsVector stack_;
    /// this index points to first redo action or equals to size of stack if no redo is available
    size_t firstRedoIndex_{ 0 };
    /// this index points to the position in stack_ corresponding to saved scene state;
    /// if firstRedoIndex_ == savedSceneIndex_ then the scene is considered as not modified
    size_t savedSceneIndex_{ 0 };
    /// memory limit (bytes) to this HistoryStore if stack_ exceed it, old actions are removed
    size_t storageLimit_{ size_t( ~0 ) };
};

/// \}

}
