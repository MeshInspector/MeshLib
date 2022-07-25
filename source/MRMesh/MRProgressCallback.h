#pragma once
#include <functional>

namespace MR
{

/// Argument value - progress in [0,1];
/// returns true to continue the operation and returns false to stop the operation
/// \ingroup BasicStructuresGroup
typedef std::function<bool( float )> ProgressCallback;

const ProgressCallback emptyProgressCallback = [] ( float )
{
    return true;
};

}
