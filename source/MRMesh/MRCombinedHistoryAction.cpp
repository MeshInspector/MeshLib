#include "MRCombinedHistoryAction.h"
#include "MRMesh/MRHistoryStore.h"

namespace MR
{



CombinedHistoryAction::CombinedHistoryAction( const std::string& name, const std::vector<std::shared_ptr<HistoryAction>>& actions ) :
    actions_{ actions },
    name_{ name }
{
}

void CombinedHistoryAction::action( HistoryAction::Type type )
{
    if ( actions_.empty() )
        return;
    if ( type == HistoryAction::Type::Undo )
    {
        for ( int i = int( actions_.size() ) - 1; i >= 0; --i )
            if ( actions_[i] )
                actions_[i]->action( type );
    }
    else if ( type == HistoryAction::Type::Redo )
    {
        for ( auto& histAct : actions_ )
            if ( histAct )
                histAct->action( type );
    }
}

bool CombinedHistoryAction::filter( HistoryStackFilter filteringCondition )
{
    return filterHistoryActionsVector( actions_, filteringCondition ).first;
}

}
