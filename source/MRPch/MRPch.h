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
#if __clang_major__ >= 18
#pragma clang diagnostic ignored "-Wenum-constexpr-conversion"
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

#include "MRJson.h"
#include "MRSpdlog.h"
#include "MRSuppressWarning.h"

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

#include "MRTBB.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <codecvt>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctype.h>
#include "MRFilesystem.h"
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <istream>
#include <iterator>
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
#include <unordered_map>
#include <variant>
#include <vector>

#ifdef MR_PCH_USE_EXTRA_HEADERS
#include "MRMesh/MRIOFilters.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRVisualObject.h"

#ifndef MESHLIB_NO_VIEWER
#include "MRViewer/MRAppendHistory.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRUIStyle.h"
#endif
#endif

#pragma warning(pop)
