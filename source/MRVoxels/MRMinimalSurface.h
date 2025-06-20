#pragma once

#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRExpected.h>
#include <MRVoxels/MRVoxelsFwd.h>


namespace MR
{


/// Supported types of TPMS (Triply Periodic Minimal Surfaces)
enum class TPMSType : int
{
    SchwartzP,
    DoubleSchwartzP,
    Gyroid,
    DoubleGyroid,
};
MRVOXELS_API std::vector<std::string> getTPMSTypeNames();

/// Construct TPMS using implicit function (https://www.researchgate.net/publication/350658078_Computational_method_and_program_for_generating_a_porous_scaffold_based_on_implicit_surfaces)
/// @param type Type of the surface
/// @param size Size of the cube with the surface
/// @param frequency Frequency of oscillations (determines size of the "cells" in the "grid")
/// @param resolution Ratio `n / T`, between the number of voxels and period of oscillations
/// @return Distance-volume starting at (0, 0, 0) and having specified @p size
MRVOXELS_API FunctionVolume buildTPMSVolume( TPMSType type, const Vector3f& size, float frequency, float resolution );

/// Constructs TPMS level-set and then convert it to mesh
MRVOXELS_API Expected<Mesh> buildTPMS( TPMSType type, const Vector3f& size, float frequency, float resolution, float iso, ProgressCallback cb = {} );

/// Constructs TPMS-filling for the given @p mesh
MRVOXELS_API Expected<Mesh> fillWithTPMS( TPMSType type, const Mesh& mesh, float frequency, float resolution, float iso, ProgressCallback cb = {} );

/// Returns number of voxels that would be used to perform \ref fillWithTPMS
MRVOXELS_API size_t getNumberOfVoxelsForTPMS( const Mesh& mesh, float frequency, float resolution );

}