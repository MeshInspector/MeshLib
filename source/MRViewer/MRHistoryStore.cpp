#include "MRHistoryStore.h"
#include "MRViewer.h"
#include "MRMesh/MRCombinedHistoryAction.h"
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

void HistoryStore::appendAction( const std::shared_ptr<HistoryAction>& action )
{
    if ( !action )
        return;
    if ( scopedBlock_ )
    {
        scopedBlock_->push_back( action );
        return;
    }
    spdlog::info( "History action append: \"{}\"", action->name() );

    stack_.resize( firstRedoIndex_ + 1 );
    stack_[firstRedoIndex_] = action;
    ++firstRedoIndex_;

    size_t currentStackSize = 0;
    for ( const auto& act : stack_ )
        currentStackSize += act->heapBytes();

    size_t numActionsToDelete = 0;
    while ( currentStackSize > storageLimit_ )
        currentStackSize -= stack_[numActionsToDelete++]->heapBytes();
    
    if ( numActionsToDelete > 0 )
    {
        stack_.erase( stack_.begin(), stack_.begin() + numActionsToDelete );
        firstRedoIndex_ -= numActionsToDelete;
        savedSceneIndex_ -= numActionsToDelete;
    }

    changedSignal( *this, ChangeType::AppendAction );
}

void HistoryStore::clear()
{
    if ( stack_.empty() )
        return;
    spdlog::info( "History store clear" );
    stack_.clear();
    firstRedoIndex_ = 0;
    changedSignal( *this, ChangeType::Clear );
}

void HistoryStore::filterStack( HistoryStackFilter filteringCondition, bool deepFiltering /*= true*/ )
{
    const auto [needSignal, redoDecrease] = filterHistoryActionsVector( stack_, filteringCondition, firstRedoIndex_, deepFiltering );
    firstRedoIndex_ -= redoDecrease;
    if ( needSignal )
        changedSignal( *this, ChangeType::Filter );
}

bool HistoryStore::undo()
{
    if ( firstRedoIndex_ == 0 )
        return false;
    assert( stack_.size() >= firstRedoIndex_ );
    if ( stack_[firstRedoIndex_ - 1] )
    {
        spdlog::info( "History action undo: \"{}\"", stack_[firstRedoIndex_ - 1]->name() );
        stack_[firstRedoIndex_ - 1]->action( HistoryAction::Type::Undo );
    }
    --firstRedoIndex_;
    changedSignal( *this, ChangeType::Undo );
    return true;
}

bool HistoryStore::redo()
{
    if ( firstRedoIndex_ >= stack_.size() )
        return false;
    if ( stack_[firstRedoIndex_] )
    {
        spdlog::info( "History action redo: \"{}\"", stack_[firstRedoIndex_]->name() );
        stack_[firstRedoIndex_]->action( HistoryAction::Type::Redo );
    }
    ++firstRedoIndex_;
    changedSignal( *this, ChangeType::Redo );
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

std::string HistoryStore::getLastActionName( HistoryAction::Type type ) const
{
    std::string res;
    std::shared_ptr<HistoryAction> action;
    if ( type == HistoryAction::Type::Undo && firstRedoIndex_ >= 1 && firstRedoIndex_ < stack_.size() + 1 )
        action = stack_[firstRedoIndex_ - 1];
    else if ( type == HistoryAction::Type::Redo && firstRedoIndex_ < stack_.size() )
        action = stack_[firstRedoIndex_];
    if ( action )
        res = action->name();
    return res;
}

}
