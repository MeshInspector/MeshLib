#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"
#include "MRExpected.h"
#include <string>

namespace MR
{
/// Mode of processing components
enum class NestedComponenetsMode
{
    Remove, ///< Default: separate nested meshes and remove them, just like union operation should do, use this if input meshes are single component
    Merge, ///< merge nested meshes, useful if input meshes are components of single object
    Union ///< does not separate components and call union for all input meshes, works slower than Remove and Merge method but returns valid result if input meshes has multiple components
};

/// Parameters structure for uniteManyMeshes function
struct UniteManyMeshesParams
{
    /// Apply random shift to each mesh, to prevent degenerations on coincident surfaces
    bool useRandomShifts{ false };
    
    /// Try fix degenerations after each boolean step, to prevent boolean failure due to high amount of degenerated faces
    /// useful on meshes with many coincident surfaces 
    /// (useRandomShifts used for same issue)
    bool fixDegenerations{ false };
    
    /// Max allowed random shifts in each direction, and max allowed deviation after degeneration fixing
    /// not used if both flags (useRandomShifts,fixDegenerations) are false
    float maxAllowedError{ 1e-5f };
    
    /// Seed that is used for random shifts
    unsigned int randomShiftsSeed{ 0 };
    
    /// If set, the bitset will store new faces created by boolean operations
    FaceBitSet* newFaces{ nullptr };

    /// By default function separate nested meshes and remove them, just like union operation should do
    /// read comment of NestedComponenetsMode enum for more information
    NestedComponenetsMode nestedComponentsMode{ NestedComponenetsMode::Remove };

    /// If set - merges meshes instead of booleaning it if boolean operation fails
    bool mergeOnFail{ false };

    /// If this option is enabled boolean will try to cut meshes even if there are self-intersections in intersecting area
    /// it might work in some cases, but in general it might prevent fast error report and lead to other errors along the way
    /// \warning not recommended in most cases
    bool forceCut = false;

    ProgressCallback progressCb;
};

// Computes the surface of objects' union each of which is defined by its own surface mesh
// - merge non intersecting meshes first
// - unite merged groups
MRMESH_API Expected<Mesh> uniteManyMeshes( const std::vector<const Mesh*>& meshes, 
    const UniteManyMeshesParams& params = {} );

}