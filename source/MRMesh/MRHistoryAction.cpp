#include "MRHistoryAction.h"
#include "MRCombinedHistoryAction.h"

namespace MR
{

std::pair<bool, int> filterHistoryActionsVector( HistoryActionsVector& historyVector,
    HistoryStackFilter filteringCondition, size_t firstRedoIndex /*= 0*/, bool deepFiltering /*= true */ )
{
    bool needSignal = false;
    int redoDecrease = 0;
    for ( int i = ( int )historyVector.size() - 1; i >= 0; --i )
    {
        if ( filteringCondition( historyVector[i] ) )
        {
            if ( i < firstRedoIndex ) ++redoDecrease;
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
                if ( i < firstRedoIndex ) ++redoDecrease;
                historyVector.erase( historyVector.begin() + i );
                needSignal = true;
            }
        }
    }
    return { needSignal, redoDecrease };
}

} //namespace MR
