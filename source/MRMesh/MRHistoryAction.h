#pragma once
#include "MRMeshFwd.h"
#include <functional>
#include <string>

namespace MR
{

/// Abstract class for history actions
/// \ingroup HistoryGroup
class HistoryAction
{
public:
    virtual ~HistoryAction() = default;

    virtual std::string name() const = 0;

    enum class Type
    {
        Undo,
        Redo
    };
    /// This function is called on history action (undo, redo, etc.)
    virtual void action( Type actionType ) = 0;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] virtual size_t heapBytes() const = 0;
};

using HistoryStackFilter = std::function<bool( const std::shared_ptr<HistoryAction>& )>;
using HistoryActionsVector = std::vector<std::shared_ptr<HistoryAction>>;

/**
 * \brief Remove actions from history actions vector that match the condition
 * \param firstRedoIndex - set redo index for calculate how many actions removed before it
 * \param deepFiltering - filter actions into combined actions
 * \return pair (anything removed, how many removed before firstRedoIndex)
 */
MRMESH_API std::pair<bool, int> filterHistoryActionsVector( HistoryActionsVector& historyVector,
    HistoryStackFilter filteringCondition, size_t firstRedoIndex = 0, bool deepFiltering = true );

} //namespace MR
