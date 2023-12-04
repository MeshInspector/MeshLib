#pragma once
#include "MRHistoryAction.h"
#include "MRObject.h"
#include "MRAffineXf3.h"
#include <memory>

namespace MR
{

/// This is an action to undo/redo a change in the value of a variable
/// \ingroup HistoryGroup
template <typename T>
class ChangeValue : public HistoryAction
{
public:
    /// Constructed from original obj
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
