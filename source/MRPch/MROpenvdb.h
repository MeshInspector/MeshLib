#pragma once

#include "MRTBB.h"

#ifndef M_PI
#define M_PI   3.1415926535897932384626433832795
#endif

#ifndef M_PI_2
#define M_PI_2 1.5707963267948966192313216916398
#endif

#pragma warning(push)

#pragma warning(disable:4005) // 'M_PI': macro redefinition
#pragma warning(disable:4146)
#pragma warning(disable:4244)
#pragma warning(disable:4251)
#pragma warning(disable:4275)
#pragma warning(disable:4127)
#pragma warning(disable:4355) //'this': used in base member initializer list
#pragma warning(disable:4464) //relative include path contains '..'
#pragma warning(disable:4701) //potentially uninitialized local variable 'inv' used

#pragma warning(disable:4868) //compiler may not enforce left-to-right evaluation order in braced initializer list

#pragma warning(disable:6297)  //Arithmetic overflow:  32-bit value is shifted, then cast to 64-bit value.  Results might not be an expected value.
#pragma warning(disable:26451) //Arithmetic overflow: Using operator '-' on a 4 byte value and then casting the result to a 8 byte value. Cast the value to the wider type before calling operator '-' to avoid overflow (io.2).
#pragma warning(disable:26495) //Variable 'openvdb::v7_1::math::Tuple<4,int>::mm' is uninitialized. Always initialize a member variable (type.6).
#pragma warning(disable:26812) //The enum type 'openvdb::v7_1::GridClass' is unscoped. Prefer 'enum class' over 'enum' (Enum.3).
#pragma warning(disable:26815) //The pointer is dangling because it points at a temporary instance which was destroyed.

// unknown pragmas
#pragma warning(disable:4068)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <openvdb/openvdb.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/Dense.h>

#pragma GCC diagnostic pop
#pragma warning(pop)

#ifdef _WIN32
//clean up after windows.h
#undef small
#undef APIENTRY
#endif
