#pragma once

#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRExpected.h>
#include <MRVoxels/MRVoxelsFwd.h>


namespace MR::TPMS // Triply Periodic Minimal Surface
{


/// Supported types of TPMS (Triply Periodic Minimal Surfaces)
enum class Type : int
{
    SchwartzP,
    DoubleSchwartzP,
    Gyroid,
    DoubleGyroid,

    Count
};
MRVOXELS_API std::vector<std::string> getTypeNames();


struct VolumeParams
{
    Type type = Type::SchwartzP;
    float frequency = 1.f;
    float resolution = 5.f;
};

struct MeshParams : VolumeParams
{
    float iso = 0.f;
};

/// Construct TPMS using implicit function (https://www.researchgate.net/publication/350658078_Computational_method_and_program_for_generating_a_porous_scaffold_based_on_implicit_surfaces)
/// @param type Type of the surface
/// @param size Size of the cube with the surface
/// @param frequency Frequency of oscillations (determines size of the "cells" in the "grid")
/// @param resolution Ratio `n / T`, between the number of voxels and period of oscillations
/// @return Distance-volume starting at (0, 0, 0) and having specified @p size
MRVOXELS_API FunctionVolume buildVolume( const VolumeParams& params, const Vector3f& size );

/// Constructs TPMS level-set and then convert it to mesh
MRVOXELS_API Expected<Mesh> build( const MeshParams& params, const Vector3f& size, ProgressCallback cb = {} );

/// Constructs TPMS-filling for the given @p mesh
MRVOXELS_API Expected<Mesh> fill( const MeshParams& params, const Mesh& mesh, ProgressCallback cb = {} );

/// Returns number of voxels that would be used to perform \ref fillWithTPMS
MRVOXELS_API size_t getNumberOfVoxels( const Mesh& mesh, float frequency, float resolution );

/// Returns number of voxels that would be used to perform \ref buildTPMS or \ref buildTPMSVolume
MRVOXELS_API size_t getNumberOfVoxels( const Vector3f& size, float frequency, float resolution );

/// Returns approximated ISO value corresponding to the given density
/// @param targetDensity value in [0; 1]
/// @return Value in [-1; 1]
MRVOXELS_API float estimateIso( Type type, float targetDensity );

/// Returns approximate density corresponding to the given ISO value
/// @param targetIso value in [-1; 1]
/// @return Value in [0; 1]
MRVOXELS_API float estimateDensity( Type type, float targetIso );

}