#pragma once

#pragma warning(push)
#pragma warning(disable: 4820) //#pragma warning: N bytes padding added after data member

#include "MREigen.h"
#include "MRHashMap.h"
#include "MRExpected.h"

#pragma warning(push)
#pragma warning(disable: 4619) // #pragma warning: there is no warning number
#pragma warning(disable: 4643) // Forward declaring 'align_val_t' in namespace std is not permitted by the C++ Standard.
#pragma warning(disable: 5204) // class has virtual functions, but its trivial destructor is not virtual; instances of objects derived from this class may not be destructed correctly
#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning(disable: 5267) // definition of implicit copy constructor is deprecated because it has a user-provided destructor
#endif
#pragma warning(disable: 5311) // A literal - operator-id of the form 'operator string-literal identifier' has been deprecated (VS2026 v18.0.0)
#if __clang_major__ >= 18
#pragma clang diagnostic ignored "-Wenum-constexpr-conversion"
#endif
#include <boost/algorithm/string.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/signals2/signal.hpp>
#include <boost/signals2/connection.hpp>
#include <boost/stacktrace.hpp>
#pragma warning(pop)

#include "MRJson.h"
#include "MRSpdlog.h"
#include "MRSuppressWarning.h"

#include "MRWinapi.h"
#ifdef _WIN32
#include <shlobj.h>
#include <commdlg.h>
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

#ifndef MESHLIB_NO_VIEWER
#ifdef __EMSCRIPTEN__
#include <GLES3/gl3.h>
#else
#include <glad/glad.h>
#endif
#include <GLFW/glfw3.h>
#endif

#include "MRStdlib.h"

#ifdef MR_PCH_USE_EXTRA_HEADERS
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRFunctional.h"
#include "MRMesh/MRIOFilters.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRParallel.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTbbThreadMutex.h"
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

// in MSVC it dramaticcaly increases the size of PCH file (from 400Mb to 800Mb),
// but greatly improves compilation time of TUs that include OpenVDB
#ifdef MR_PCH_USE_OPENVDB
#include "MRVoxels/MROpenVDB.h"
#endif

#pragma warning(pop)
