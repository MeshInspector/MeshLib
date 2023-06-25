#include "MRExpected.h"

#ifdef __cpp_lib_expected
#pragma message("std::expected from C++23 is available")
#else
#pragma message("std::expected is NOT available, using tl::expected instead")
#endif
