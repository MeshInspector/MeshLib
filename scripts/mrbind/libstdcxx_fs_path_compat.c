#include <stdint.h>

// std::filesystem::__cxx11::path::_List::type(_Type) per GCC 11/12 fs_path.cc: tagged-pointer math on
// _List's sole member (a unique_ptr whose low 2 bits carry _Type). Clang references it as extern when
// compiling against those headers, but EL8 defines it in no library; linked in via EXTRA_LDFLAGS.
void _ZNSt10filesystem7__cxx114path5_List4typeENS1_5_TypeE(uintptr_t* self, unsigned char t)
{
    *self = (*self & ~(uintptr_t)3) | t;
}
