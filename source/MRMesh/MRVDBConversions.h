#pragma once

#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRMeshFwd.h"
#include "MRMeshPart.h"
#include "MRProgressCallback.h"
#include "MRAffineXf3.h"
#include <tl/expected.hpp>
#include <string>

namespace MR
{
// closed surface is required
// surfaceOffset - number voxels around surface to calculate distance in (should be positive)
// returns null if was canceled by progress callback
MRMESH_API FloatGrid meshToLevelSet( const MeshPart& mp, const AffineXf3f& xf,
                                     const Vector3f& voxelSize, float surfaceOffset = 3,
                                     ProgressCallback cb = {} );

// does not require closed surface, resulting grid cannot be used for boolean operations,
// surfaceOffset - the number of voxels around surface to calculate distance in (should be positive)
// returns null if was canceled by progress callback
MRMESH_API FloatGrid meshToDistanceField( const MeshPart& mp, const AffineXf3f& xf,
                                          const Vector3f& voxelSize, float surfaceOffset = 3,
                                          ProgressCallback cb = {} );

// Parameters structure for meshToVolume function
struct MeshToVolumeParams
{
    // Conversion type
    enum class Type
    {
        Signed, // only closed meshes can be converted with signed type
        Unsigned // this type leads to shell like iso-surfaces
    } type{ Type::Unsigned };
    float surfaceOffset{ 3.0 }; // the number of voxels around surface to calculate distance in (should be positive)
    Vector3f voxelSize = Vector3f::diagonal( 1.0f );
    AffineXf3f worldXf; // mesh initial transform
    AffineXf3f* outXf{ nullptr }; // optional output: xf to original mesh (respecting worldXf)
    ProgressCallback cb;
};

// convert mesh to volume in (0,0,0)-(dim.x,dim.y,dim.z) grid box
MRMESH_API tl::expected<VdbVolume, std::string> meshToVolume( const Mesh& mesh, const MeshToVolumeParams& params = {} );

// fills VdbVolume data from FlaotGrid (does not fill voxels size, cause we expect it outside)
MRMESH_API VdbVolume floatGridToVdbVolume( const FloatGrid& grid );

// make FloatGrid from SimpleVolume
// make copy of data
// grid can be used to make iso-surface later with gridToMesh function
MRMESH_API FloatGrid simpleVolumeToDenseGrid( const SimpleVolume& simpleVolume, ProgressCallback cb = {} );
MRMESH_API VdbVolume simpleVolumeToVdbVolume( const SimpleVolume& simpleVolume, ProgressCallback cb = {} );

// isoValue - layer of grid with this value would be converted in mesh
// isoValue can be negative only in level set grids
// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones 
//                       (curvature can be lost on high values)
MRMESH_API tl::expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
    float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );
MRMESH_API tl::expected<Mesh, std::string> gridToMesh( const VdbVolume& vdbVolume,
    float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );

// isoValue - layer of grid with this value would be converted in mesh
// isoValue can be negative only in level set grids
// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones 
//                       (curvature can be lost on high values)
// maxFaces if mesh faces exceed this value error returns
MRMESH_API tl::expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
    int maxFaces, float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );
MRMESH_API tl::expected<Mesh, std::string> gridToMesh( const VdbVolume& vdbVolume, int maxFaces,
    float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );

// performs convention from mesh to levelSet and back with offsetA, and than same with offsetB
// allowed only for closed meshes
// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones 
//                       (curvature can be lost on high values)
MRMESH_API tl::expected<Mesh, std::string> levelSetDoubleConvertion( const MeshPart& mp, const AffineXf3f& xf,
    float voxelSize, float offsetA, float offsetB, float adaptivity = 0.0f, ProgressCallback cb = {} );

}
#endif
