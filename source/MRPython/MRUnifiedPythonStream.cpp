#include "MRUnifiedPythonStream.h"

namespace MR
{
std::stringstream& UnifiedPythonStream::get()
{
    static UnifiedPythonStream self;
    return self.ss_;
}
}
