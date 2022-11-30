#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"
#include <tl/expected.hpp>
#include <string>

namespace MR
{

// Parameters structure for uniteManyMeshes function
struct UniteManyMeshesParams
{
    // Apply random shift to each mesh, to prevent degenerations on coincident surfaces
    bool useRandomShifts{ false };
    // Try fix degenerations after each boolean step, to prevent boolean failure due to high amount of degenerated faces
    // useful on meshes with many coincident surfaces 
    // (useRandomShifts used for same issue)
    bool fixDegenerations{ false };
    // Max allowed random shifts in each direction, and max allowed deviation after degeneration fixing
    // not used if both flags (useRandomShifts,fixDegenerations) are false
    float maxAllowedError{ 1e-5f };
    // Seed that is used for random shifts
    unsigned int randomShiftsSeed{ 0 };
    // If set, the bitset will store new faces created by boolean operations
    FaceBitSet* newFaces{ nullptr };
};

// Computes the surface of objects' union each of which is defined by its own surface mesh
// - merge non intersecting meshes first
// - unite merged groups
MRMESH_API tl::expected<Mesh, std::string> uniteManyMeshes( const std::vector<const Mesh*>& meshes, 
    const UniteManyMeshesParams& params = {} );

}