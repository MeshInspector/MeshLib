#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// the class to compute the volume of water some basin can accumulate,
/// considering that water upper surface has constant z-level
class BasinVolumeCalculator
{
public:
    /// pass every triangle of the basin here, and the water level;
    /// \return true if the triangle is at least partially below the water level and influences on the volume
    MRMESH_API bool addTerrainTri( Triangle3f t, float level );

    /// call it after all addTerrainTri to get the volume
    [[nodiscard]] double getVolume() const { return sum_ / 6; }

private:
    double sum_ = 0;
};

/// computes the volume of given mesh basin below given water level;
/// \param faces shall include all basin faces at least partially below the water level
[[nodiscard]] MRMESH_API double computeBasinVolume( const Mesh& mesh, const FaceBitSet& faces, float level );

} //namespace MR
