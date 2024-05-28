#pragma once

#pragma warning(push)
#pragma warning(disable: 4820) //#pragma warning: N bytes padding added after data member

#include "MREigen.h"

#include <parallel_hashmap/phmap.h>

#include <version>
#ifdef __cpp_lib_expected
#include <expected>
#else
#include <tl/expected.hpp>
#endif

#pragma warning(push)
#pragma warning(disable: 4619) //#pragma warning: there is no warning number
#pragma warning(disable: 4643) //Forward declaring in namespace std is not permitted by the C++ Standard.
#pragma warning(disable: 5204) //class has virtual functions, but its trivial destructor is not virtual; instances of objects derived from this class may not be destructed correctly
#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning(disable: 5267) //definition of implicit copy constructor is deprecated because it has a user-provided destructor
#endif
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#include <boost/algorithm/string.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/signals2/signal.hpp>
#include <boost/stacktrace.hpp>
#pragma warning(pop)

#ifndef __EMSCRIPTEN__
#pragma warning(push)
#pragma warning(disable: 4515)
#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning(disable: 5267) //definition of implicit copy constructor is deprecated because it has a user-provided destructor
#endif
#include <gdcmImageReader.h>
#include <gdcmImageHelper.h>
#pragma warning(pop)

//adding this include decreases MI compilation time from 160 sec to 130 sec at the cost of twice large MRPch.pch (436M -> 903M)
//#include "MROpenvdb.h"
#endif

#include "MRJson.h"
#include "MRSpdlog.h"
#include "MRSuppressWarning.h"

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#include <Commdlg.h>
#endif

#ifndef __EMSCRIPTEN__
#include <fmt/chrono.h>
#endif

#include <gtest/gtest.h>

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4355) //'this': used in base member initializer list
#pragma warning(disable:5204) //class has virtual functions, but its trivial destructor is not virtual; instances of objects derived from this class may not be destructed correctly
#pragma warning(disable:5220) //a non-static data member with a volatile qualified type no longer implies that compiler generated copy/move constructors and copy/move assignment operators are not trivial
#include <ppltasks.h>
#pragma warning(pop)
#endif

#if !defined( __EMSCRIPTEN__)
#pragma warning(push)
#pragma warning(disable:4244) //'initializing': conversion from 'std::streamoff' to 'int', possible loss of data
#pragma warning(disable:4355) //'this': used in base member initializer list
#pragma warning(disable:4265) //class has virtual functions, but its non-trivial destructor is not virtual; instances of this class may not be destructed correctly
#pragma warning(disable:4458) //'this': declaration of 'size' hides class member
#include <cpr/cpr.h>
#pragma warning(pop)
#endif

#ifndef __EMSCRIPTEN__
#include <libpng16/png.h>
#endif

#include "OpenCTM/openctm.h"

#if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
// in Debug Clang builds with PCH including pybind, all libraries and executable depend on python
#if !defined(__clang__) || defined(NDEBUG)
#pragma warning(push)
#pragma warning(disable:4100) //'_unused_op': unreferenced formal parameter
#pragma warning(disable:4189) //'has_args': local variable is initialized but not referenced
#pragma warning(disable:4191) //'reinterpret_cast': unsafe conversion from 'PyObject *(__cdecl *)(PyObject *,PyObject *,PyObject *)' to 'void (__cdecl *)(void)'
#pragma warning(disable:4355) //'this': used in base member initializer list
#pragma warning(disable:4464) //relative include path contains '..'
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#pragma warning(pop)
#endif
#endif

#include "MRTBB.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <codecvt>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctype.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <istream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <queue>
#include <string>
#include <thread>
#include <tuple>
#include <variant>
#include <vector>
#include <unordered_map>

#pragma warning(pop)
