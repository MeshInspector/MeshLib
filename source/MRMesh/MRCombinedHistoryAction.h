#pragma once
#include "MRMeshFwd.h"
#include "MRHistoryAction.h"
#include <memory>
#include <optional>

namespace MR
{

/// History action for combine some history actions
/// \ingroup HistoryGroup
class MRMESH_CLASS CombinedHistoryAction : public HistoryAction
{
public:
    /// A function that generates and returns the action's name on each call.
    /// Useful when the correct action name depends on the current conditions, e.g. the current UI language.
    using DynamicNameGetter = std::function<std::string ()>;

    /// Will call action() for each actions in given order (undo in reverse, redo in forward)
    MRMESH_API CombinedHistoryAction( const std::string& name, const std::vector<std::shared_ptr<HistoryAction>>& actions, const DynamicNameGetter& dynNameGetter = {} );

    virtual std::string name() const override
    {
        return name_;
    }

    MRMESH_API virtual void action( HistoryAction::Type type ) override;

          HistoryActionsVector& getStack()       { return actions_; }
    const HistoryActionsVector& getStack() const { return actions_; }

    /// Remove some actions according to condition inside combined actions.
    /// Do deep filtering.
    MRMESH_API bool filter( HistoryStackFilter filteringCondition );

    bool empty() const { return actions_.empty(); };

    /// For actions with a dynamic name getter calls the getter and returns the dynamic name; otherwise returns std::nullopt.
    MRMESH_API std::optional<std::string> getDynamicName() const;

    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

private:
    HistoryActionsVector actions_;
    std::string name_;
    DynamicNameGetter dynNameGetter_;
};

/// Returns dynamic name if the action is an instance of CombinedHistoryAction; otherwise returns std::nullopt.
MRMESH_API std::optional<std::string> getDynamicName( const std::shared_ptr<HistoryAction>& action );

}
