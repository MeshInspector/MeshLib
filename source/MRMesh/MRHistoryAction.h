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

}