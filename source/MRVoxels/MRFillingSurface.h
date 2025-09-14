#pragma once

#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRExpected.h>
#include <MRMesh/MRVector3.h>
#include <MRVoxels/MRVoxelsFwd.h>

#include <variant>
#include <optional>

namespace MR::FillingSurface
{

namespace TPMS // Triply Periodic Minimal Surface
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

} // namespace TPMS



namespace CellularSurface // Surface of cylinders in a grid
{

struct Params
{
    Vector3f period = Vector3f::diagonal( 1.f );    ///< the distance between consecutive cylinders in each direction
    Vector3f width = Vector3f::diagonal( 0.3f );    ///< the width of cylinders in each direction
    float r = 0.4f;         ///< the radius of uniting spheres

    // used in tests in order to make surfaces close to their analytical expression
    // recommended to be false for real usage for better performance
    bool highRes = false;

};

/// Build a cellular surface of size \p size
MRVOXELS_API Expected<Mesh> build( const Vector3f& size, const Params& params, const ProgressCallback& cb = {} );

/// Fill given mesh with a cellular surface
MRVOXELS_API Expected<Mesh> fill( const Mesh& mesh, const Params& params, const ProgressCallback& cb = {} );

/// Estimate the density of the cellular surface
MRVOXELS_API float estimateDensity( float period, float width, float r );

/// Estimate the width that is needed to attain the \p targetDensity. Inverse of \ref estimateDensity.
/// \note The width is not unique in general, no guarantees are made about which value among possible will be returned.
//    Due to the simplification of the formula (sphere must either fully contain the intersection of cylinders or be inside it), solution not always exists.
MRVOXELS_API std::optional<float> estimateWidth( float period, float r, float targetDensity );

}


// Different kinds of filling surface
enum class Kind : int
{
    TPMS = 0,
    Cellular
};
MRVOXELS_API std::vector<std::string> getKindNames();

using MeshParamsRef = std::variant
    < std::reference_wrapper<TPMS::MeshParams>
    , std::reference_wrapper<CellularSurface::Params>
    >;

using ConstMeshParamsRef = std::variant
    < std::reference_wrapper<const TPMS::MeshParams>
    , std::reference_wrapper<const CellularSurface::Params>
    >;

/// Unified functions to build and fill using the specified filling structures.
MR_BIND_IGNORE MRVOXELS_API Expected<Mesh> build( const Vector3f& size, ConstMeshParamsRef params, ProgressCallback cb = {} );
MR_BIND_IGNORE MRVOXELS_API Expected<Mesh> fill( const Mesh& mesh, ConstMeshParamsRef params, ProgressCallback cb = {} );

} // namespace FillingSurface