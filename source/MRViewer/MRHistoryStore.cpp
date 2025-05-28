#include "MRHistoryStore.h"
#include "MRViewer.h"
#include "MRMesh/MRCombinedHistoryAction.h"
#include "MRMesh/MRFinally.h"
#include "MRPch/MRSpdlog.h"
#include <cassert>

namespace MR
{

const std::shared_ptr<HistoryStore>& HistoryStore::getViewerInstance()
{
    return MR::getViewerInstance().getGlobalHistoryStore();
}

HistoryStore::~HistoryStore()
{
    clear();
}

void HistoryStore::appendAction( std::shared_ptr<HistoryAction> action )
{
    assert( !undoRedoInProgress_ );
    if ( undoRedoInProgress_ )
        return;
    if ( !action )
        return;
    if ( scopedBlock_ )
    {
        scopedBlock_->push_back( std::move( action ) );
        return;
    }
    spdlog::info( "History action append: \"{}\"", action->name() );
    assert( !action->name().empty() );

    stack_.resize( firstRedoIndex_ + 1 );
    stack_[firstRedoIndex_] = std::move( action );
    ++firstRedoIndex_;

    changedSignal( *this, ChangeType::AppendAction );

    filterByMemoryLimit_();
}

void HistoryStore::clear()
{
    assert( !undoRedoInProgress_ );
    if ( undoRedoInProgress_ )
        return;
    if ( stack_.empty() )
        return;
    spdlog::info( "History store clear" );
    stack_.clear();
    firstRedoIndex_ = 0;
    changedSignal( *this, ChangeType::Clear );
}

void HistoryStore::filterStack( HistoryStackFilter filteringCondition, bool deepFiltering /*= true*/ )
{
    assert( !undoRedoInProgress_ );
    if ( undoRedoInProgress_ )
        return;
    const auto [needSignal, redoDecrease] = filterHistoryActionsVector( stack_, filteringCondition, firstRedoIndex_, deepFiltering );
    firstRedoIndex_ -= redoDecrease;
    if ( needSignal )
        changedSignal( *this, ChangeType::Filter );
}

bool HistoryStore::undo()
{
    assert( !undoRedoInProgress_ );
    if ( undoRedoInProgress_ )
        return false;
    if ( firstRedoIndex_ == 0 )
        return false;

    undoRedoInProgress_ = true;
    MR_FINALLY { undoRedoInProgress_ = false; };
    assert( stack_.size() >= firstRedoIndex_ );
    if ( stack_[firstRedoIndex_ - 1] )
    {
        spdlog::info( "History action undo: \"{}\"", stack_[firstRedoIndex_ - 1]->name() );
        changedSignal( *this, ChangeType::PreUndo );
        stack_[firstRedoIndex_ - 1]->action( HistoryAction::Type::Undo );
    }
    --firstRedoIndex_;
    changedSignal( *this, ChangeType::PostUndo );
    return true;
}

bool HistoryStore::redo()
{
    assert( !undoRedoInProgress_ );
    if ( undoRedoInProgress_ )
        return false;
    if ( firstRedoIndex_ >= stack_.size() )
        return false;

    undoRedoInProgress_ = true;
    MR_FINALLY { undoRedoInProgress_ = false; };
    if ( stack_[firstRedoIndex_] )
    {
        spdlog::info( "History action redo: \"{}\"", stack_[firstRedoIndex_]->name() );
        changedSignal( *this, ChangeType::PreRedo );
        stack_[firstRedoIndex_]->action( HistoryAction::Type::Redo );
    }
    ++firstRedoIndex_;
    changedSignal( *this, ChangeType::PostRedo );
    return true;
}

std::vector<std::string> HistoryStore::getNActions( unsigned n, HistoryAction::Type type ) const
{
    if ( type == HistoryAction::Type::Undo )
        n = std::min( unsigned( firstRedoIndex_ ), n );
    else if ( type == HistoryAction::Type::Redo )
        n = std::min( unsigned( stack_.size() - firstRedoIndex_ ), n );
    std::vector<std::string> res( n );
    for ( unsigned i = 0; i < n; ++i )
    {
        std::shared_ptr<HistoryAction> action;
        if ( type == HistoryAction::Type::Undo )
            action = stack_[firstRedoIndex_ - 1 - i];
        else if ( type == HistoryAction::Type::Redo )
            action = stack_[firstRedoIndex_ + i];
        if ( action )
            res[i] = action->name();
    }
    return res;
}

std::shared_ptr<HistoryAction> HistoryStore::getLastAction( HistoryAction::Type type ) const
{
    std::shared_ptr<HistoryAction> action;
    if ( type == HistoryAction::Type::Undo && firstRedoIndex_ >= 1 && firstRedoIndex_ < stack_.size() + 1 )
        action = stack_[firstRedoIndex_ - 1];
    else if ( type == HistoryAction::Type::Redo && firstRedoIndex_ < stack_.size() )
        action = stack_[firstRedoIndex_];
    return action;
}

std::string HistoryStore::getLastActionName( HistoryAction::Type type ) const
{
    std::string res;
    if ( auto action = getLastAction( type ) )
        res = action->name();
    return res;
}

size_t HistoryStore::calcUsedMemory() const
{
    size_t currentStackSize = 0;
    for ( int i = 0; i < firstRedoIndex_; ++i )
        currentStackSize += stack_[i]->heapBytes();
    return currentStackSize;
}

void HistoryStore::filterByMemoryLimit_()
{
    size_t currentStackSize = calcUsedMemory();
    size_t numActionsToDelete = 0;
    while ( currentStackSize > storageLimit_ && numActionsToDelete <= firstRedoIndex_ )
        currentStackSize -= stack_[numActionsToDelete++]->heapBytes();

    for ( int i = 0; i < numActionsToDelete; ++i )
    {
        stack_.erase( stack_.begin() );
        --firstRedoIndex_;
        --savedSceneIndex_;
        changedSignal( *this, ChangeType::PopAction );
    }
}

} //namespace MR
