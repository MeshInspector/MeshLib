#pragma once

#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 4127) // conditional expression is constant
#pragma warning(disable: 4464) // relative include path contains '..'
#pragma warning(disable: 4643) // Forward declaring 'tuple' in namespace std is not permitted by the C++ Standard.
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#if __GNUC__ >= 12
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
#endif
#include <Eigen/Core>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#pragma warning(pop)

