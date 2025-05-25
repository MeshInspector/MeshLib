#include "MRAppendHistory.h"
#include "MRHistoryStore.h"
#include "MRViewer.h"
#include "MRMesh/MRCombinedHistoryAction.h"

namespace MR
{


void FilterHistoryByCondition( HistoryStackFilter condition, bool deepFiltering /*= true*/ )
{
    auto& viewer = getViewerInstance();
    auto store = viewer.getGlobalHistoryStore();
    if ( !store )
        return;
    store->filterStack( condition, deepFiltering );
}

ScopeHistory::ScopeHistory( const std::string& name )
{
    auto viewer = Viewer::instance();
    if ( !viewer )
        return;
    store_ = viewer->getGlobalHistoryStore();
    if ( !store_ )
        return;
    parentScopePtr_ = store_->getScopeBlockPtr();
    combinedAction_ = std::make_shared<CombinedHistoryAction>( name, HistoryActionsVector{} );
    store_->setScopeBlockPtr( &combinedAction_->getStack() );
}

ScopeHistory::~ScopeHistory()
{
    if ( !store_ )
        return;
    store_->setScopeBlockPtr( parentScopePtr_ );
    parentScopePtr_ = nullptr;
    if ( combinedAction_ && !combinedAction_->getStack().empty() )
        store_->appendAction( std::move( combinedAction_ ) );
}

} //namespace MR
