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
    ThickSchwartzP,
    DoubleGyroid,
    ThickGyroid,

    Count
};

/// Returns the names for each type of filling
MRVOXELS_API std::vector<std::string> getTypeNames();

/// Returns the tooltips for each type of filling
MRVOXELS_API std::vector<std::string> getTypeTooltips();

/// Returns true if the \p type is thick
MRVOXELS_API bool isThick( Type type );


struct VolumeParams
{
    Type type = Type::SchwartzP; // Type of the surface
    float frequency = 1.f; // Frequency of oscillations (determines size of the "cells" in the "grid")
    float resolution = 5.f; // Ratio `n / T`, between the number of voxels and period of oscillations
};

struct MeshParams : VolumeParams
{
    float iso = 0.f;
    bool decimate = false;
};

/// Construct TPMS using implicit function (https://www.researchgate.net/publication/350658078_Computational_method_and_program_for_generating_a_porous_scaffold_based_on_implicit_surfaces)
/// @param size Size of the cube with the surface
/// @return Distance-volume starting at (0, 0, 0) and having specified @p size
MRVOXELS_API FunctionVolume buildVolume( const Vector3f& size, const VolumeParams& params );

/// Constructs TPMS level-set and then convert it to mesh
MRVOXELS_API Expected<Mesh> build( const Vector3f& size, const MeshParams& params, ProgressCallback cb = {} );

/// Constructs TPMS-filling for the given @p mesh
MRVOXELS_API Expected<Mesh> fill( const Mesh& mesh, const MeshParams& params, ProgressCallback cb = {} );

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

/// Returns minimal reasonable resolution for given parameters
MRVOXELS_API float getMinimalResolution( Type type, float iso );

}