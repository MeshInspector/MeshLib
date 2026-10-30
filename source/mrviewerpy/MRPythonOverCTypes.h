#pragma once

// Force-included into the `mrviewerpy_over_c` target: makes `mrviewerpy` take and return the objects of the `mrmeshpy`
//   generated over the C bindings (`scripts/mrbind/python_over_c.sh`), through its C++ interop API.

#define MRBIND_PY_CPP_API_CAPSULE "meshlib.mrmeshpy.__mrbind_cpp_api__"
#include <mrbind/python_over_c/pybind11_casters.h>
#include <pybind11/stl.h>

#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRViewportId.h"
#ifndef MESHLIB_NO_VOXELS
#include "MRVoxels/MRVoxelsVolume.h"
#endif

MRBIND_PY_FOREIGN_TYPE( MR::Vector2f, "MR::Vector2f", "meshlib.mrmeshpy.Vector2f" )
MRBIND_PY_FOREIGN_TYPE( MR::Vector3f, "MR::Vector3f", "meshlib.mrmeshpy.Vector3f" )
MRBIND_PY_FOREIGN_TYPE( MR::AffineXf3f, "MR::AffineXf3f", "meshlib.mrmeshpy.AffineXf3f" )
MRBIND_PY_FOREIGN_TYPE( MR::ViewportId, "MR::ViewportId", "meshlib.mrmeshpy.ViewportId" )
MRBIND_PY_FOREIGN_TYPE( MR::ViewportMask, "MR::ViewportMask", "meshlib.mrmeshpy.ViewportMask" )
MRBIND_PY_FOREIGN_TYPE( MR::FaceBitSet, "MR::FaceBitSet", "meshlib.mrmeshpy.FaceBitSet" )
MRBIND_PY_FOREIGN_TYPE( MR::VertBitSet, "MR::VertBitSet", "meshlib.mrmeshpy.VertBitSet" )
MRBIND_PY_FOREIGN_TYPE( MR::UndirectedEdgeBitSet, "MR::UndirectedEdgeBitSet", "meshlib.mrmeshpy.UndirectedEdgeBitSet" )
MRBIND_PY_FOREIGN_TYPE( MR::Mesh, "MR::Mesh", "meshlib.mrmeshpy.Mesh" )
MRBIND_PY_FOREIGN_TYPE( MR::PointCloud, "MR::PointCloud", "meshlib.mrmeshpy.PointCloud" )
MRBIND_PY_FOREIGN_TYPE( MR::Polyline3, "MR::Polyline3", "meshlib.mrmeshpy.Polyline3" )
MRBIND_PY_FOREIGN_TYPE( MR::DistanceMap, "MR::DistanceMap", "meshlib.mrmeshpy.DistanceMap" )
MRBIND_PY_FOREIGN_TYPE( MR::Image, "MR::Image", "meshlib.mrmeshpy.Image" )
MRBIND_PY_FOREIGN_TYPE( MR::Object, "MR::Object", "meshlib.mrmeshpy.Object" )
#ifndef MESHLIB_NO_VOXELS
MRBIND_PY_FOREIGN_TYPE( MR::VdbVolume, "MR::VdbVolume", "meshlib.mrmeshpy.VdbVolume" )
#endif
