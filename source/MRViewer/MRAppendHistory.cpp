#include "MRAppendHistory.h"
#include "MRMesh/MRHistoryStore.h"
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

ScopeHistory::ScopeHistory( const std::string& name ) :
    name_{ name }
{
    auto viewer = Viewer::instance();
    if ( !viewer )
        return;
    store_ = viewer->getGlobalHistoryStore();
    if ( !store_ )
        return;
    if ( store_->isInScopeMode() )
        return;
    store_->startScope( true );
    thisScopeStartedScopedMode_ = true;
}

ScopeHistory::~ScopeHistory()
{
    if ( !thisScopeStartedScopedMode_ )
        return;
    auto scopeBlock = store_->getScopeBlock();
    store_->startScope( false );
    if ( !scopeBlock.empty() )
        AppendHistory<CombinedHistoryAction>( name_, scopeBlock );
}

}
