#pragma once
#include "MRMeshFwd.h"
#include <functional>
#include <string>

namespace MR
{
// Abstract class for history actions
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
    // This function is called on history action (undo, redo, etc.)
    virtual void action( Type actionType ) = 0;
};

using HistoryStackFilter = std::function<bool( const std::shared_ptr<HistoryAction>& )>;
using HistoryActionsVector = std::vector<std::shared_ptr<HistoryAction>>;

}