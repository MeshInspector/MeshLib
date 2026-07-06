#include <stdint.h>

// std::filesystem::__cxx11::path::_List::type(_Type) per GCC 11/12 fs_path.cc: tagged-pointer math on
// _List's sole member (a unique_ptr whose low 2 bits carry _Type). Works around https://gcc.gnu.org/PR108636
// (fixed in 11.4/12.3, EL8 ships 11.2.1/12.2.1): Clang emits an extern ref, but EL8 defines it in no library.
void _ZNSt10filesystem7__cxx114path5_List4typeENS1_5_TypeE(uintptr_t* self, unsigned char t)
{
    *self = (*self & ~(uintptr_t)3) | t;
}
