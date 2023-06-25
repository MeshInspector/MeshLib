#if defined __cpp_lib_expected
#pragma message("std::expected from C++23 is avaibled")
#else
#pragma message("std::expected is NOT avaibled, using tl::expected instead")
#endif
