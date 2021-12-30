#pragma once

#pragma warning(push)
#pragma warning(disable: 4820) //#pragma warning: N bytes padding added after data member

#include "MREigen.h"

#include <parallel_hashmap/phmap.h>

#include <tl/expected.hpp>

#pragma warning(push)
#pragma warning(disable: 4619) //#pragma warning: there is no warning number
#pragma warning(disable: 4643) //Forward declaring in namespace std is not permitted by the C++ Standard.
#pragma warning(disable: 5204) //class has virtual functions, but its trivial destructor is not virtual; instances of objects derived from this class may not be destructed correctly
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#include <boost/algorithm/string.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/signals2/signal.hpp>
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable: 4515)
#include <gdcmImageReader.h>
#include <gdcmImageHelper.h>
#pragma warning(pop)

#include "MROpenvdb.h"
#include "MRJson.h"

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#include <Commdlg.h>
#endif

#pragma warning(push)
#pragma warning(disable:4275)
#pragma warning(disable:4251)
#pragma warning(disable:4273)
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#ifdef _WIN32
#include <spdlog/sinks/msvc_sink.h>
#endif
#pragma warning(pop)

#include <fmt/chrono.h>

#include <gtest/gtest.h>

#pragma warning(push)
#pragma warning(disable:4355) //'this': used in base member initializer list
#pragma warning(disable:5204) //class has virtual functions, but its trivial destructor is not virtual; instances of objects derived from this class may not be destructed correctly
#pragma warning(disable:5220) //a non-static data member with a volatile qualified type no longer implies that compiler generated copy/move constructors and copy/move assignment operators are not trivial
#include <ppltasks.h>
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable:4355) //'this': used in base member initializer list
#pragma warning(disable:4265) //class has virtual functions, but its non-trivial destructor is not virtual; instances of this class may not be destructed correctly
#pragma warning(disable:4458) //'this': declaration of 'size' hides class member
#include <cpr/cpr.h>
#pragma warning(pop)

#include <libpng16/png.h>

#include "OpenCTM/openctm.h"

#pragma warning(push)
#pragma warning(disable:4191) //'reinterpret_cast': unsafe conversion from 'PyObject *(__cdecl *)(PyObject *,PyObject *,PyObject *)' to 'void (__cdecl *)(void)'
#pragma warning(disable:4355) //'this': used in base member initializer list
#pragma warning(disable:4464) //relative include path contains '..'
#pragma warning(disable:4686) //'pybind11::detail::descr<10,T,pybind11::str>::types': possible change in behavior, change in UDT return calling convention
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#pragma warning(pop)

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <codecvt>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
#include <vector>
#include <unordered_map>

#pragma warning(pop)
