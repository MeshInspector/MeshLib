#pragma once
#include "MRHistoryAction.h"
#include "MRObject.h"
#include "MRAffineXf3.h"
#include <memory>

namespace MR
{

/// History action for xf change
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
        auto tmpValue = *currentValue_;
        *currentValue_ = oldValue_;
        oldValue_ = tmpValue;
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::string name_;
    T* currentValue_;
    T oldValue_;
};

}
