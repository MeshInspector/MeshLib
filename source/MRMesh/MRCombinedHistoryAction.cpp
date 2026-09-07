#include "MRCombinedHistoryAction.h"
#include "MRHeapBytes.h"

namespace MR
{

CombinedHistoryAction::CombinedHistoryAction( const std::string& name, const std::vector<std::shared_ptr<HistoryAction>>& actions, const DynamicNameGetter& dynamicNameFunc )
    : actions_{ actions }
    , name_{ name }
    , dynNameGetter_{ dynamicNameFunc }
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

std::optional<std::string> CombinedHistoryAction::getDynamicName() const
{
    if ( !dynNameGetter_ )
        return {};
    return dynNameGetter_();
}

size_t CombinedHistoryAction::heapBytes() const
{
    auto res = name_.capacity() + MR::heapBytes( actions_ );
    for ( const auto & a : actions_ )
        res += MR::heapBytes( a );
    return res;
}

std::optional<std::string> getDynamicName( const std::shared_ptr<HistoryAction>& action )
{
    if ( auto combHist = std::dynamic_pointer_cast<CombinedHistoryAction>( action ) )
        return combHist->getDynamicName();
    return {};
}

}
