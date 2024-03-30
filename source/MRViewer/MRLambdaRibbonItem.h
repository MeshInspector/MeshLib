#pragma once

#include "MRRibbonMenuItem.h"

namespace MR
{

// Simple ribbon item acting given lambda
class LambdaRibbonItem : public RibbonMenuItem
{
public:
    using SimpleLambda = std::function<void()>;
    LambdaRibbonItem( std::string name, SimpleLambda lambda ) :
        RibbonMenuItem( std::move( name ) ),
        lambda_( std::move( lambda ) )
    {}

    virtual bool action() override
    {
        lambda_();
        return false;
    }
private:
    SimpleLambda lambda_;
};

} // namespace MR
