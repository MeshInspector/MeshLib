#pragma once
#include "exports.h"
#include <sstream>

namespace MR
{
// python output duplicated to this stream
// needed to be able to separate python output from all other logs
class MRPYTHON_CLASS UnifiedPythonStream
{
public:
    MRPYTHON_API static std::stringstream& get();
private:
    UnifiedPythonStream() = default;
    ~UnifiedPythonStream() = default;

    std::stringstream ss_;
};
}