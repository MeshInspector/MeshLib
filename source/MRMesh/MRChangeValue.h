#pragma once
#include "MRHistoryAction.h"
#include "MRAffineXf3.h"
#include <memory>

namespace MR
{

/// This is an action to undo/redo a change in the value of a variable
/// in the case of temporary objects, 
/// it is necessary to delete the history. 
/// this module should not exist anymore than the object that was passed to it.
/// \ingroup HistoryGroup
template <typename T>
class ChangeValue : public HistoryAction
{
public:
    /// Constructed from original object's pointer and old value
    ChangeValue( const std::string& name, T* currentValue, T oldValue ) :
        name_{ name },
        currentValue_{ currentValue },
        oldValue_{ oldValue }
    {}

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !currentValue_ )
            return;
        std::swap( *currentValue_, oldValue_ );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::string name_;
    T* currentValue_ = nullptr;
    T oldValue_;
};

}
