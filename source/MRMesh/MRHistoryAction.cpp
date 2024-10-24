#include "MRHistoryAction.h"
#include "MRCombinedHistoryAction.h"

namespace MR
{

bool filterHistoryActionsVector( HistoryActionsVector& historyVector,
    HistoryStackFilter filteringCondition, bool deepFiltering /*= true */, std::function<void( size_t id )> onFilterCb )
{
    bool needSignal = false;
    for ( int i = ( int )historyVector.size() - 1; i >= 0; --i )
    {
        if ( filteringCondition( historyVector[i] ) )
        {
            if ( onFilterCb ) onFilterCb( i );
            historyVector.erase( historyVector.begin() + i );
            needSignal = true;
        }
        else if ( deepFiltering )
        {
            auto combinedAction = std::dynamic_pointer_cast< CombinedHistoryAction >( historyVector[i] );
            if ( !combinedAction )
                continue;

            needSignal = combinedAction->filter( filteringCondition ) || needSignal;
            if ( combinedAction->empty() )
            {
                if ( onFilterCb ) onFilterCb( i );
                historyVector.erase( historyVector.begin() + i );
                needSignal = true;
            }
        }
    }
    return needSignal;
}

} //namespace MR
