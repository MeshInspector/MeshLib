#include "MRExpected.h"

#if __cpp_lib_expected >= 202211
  #pragma message("std::expected with monadic functions from C++23 is available")
#else
  #ifdef __cpp_lib_expected
    #pragma message("std::expected available but WITHOUT monadic functions, using tl::expected instead")
  #else
    #pragma message("std::expected is NOT available, using tl::expected instead")
  #endif
#endif
