#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{

/// parameters for MR::findOverhangs
struct FindOverhangsSettings
{
    /// base axis marking the up direction
    Vector3f axis;
    /// height of a layer
    float layerHeight { 1.f };
    /// maximum overhang distance within a layer
    float maxOverhangDistance { 1.f };
    /// number of hops used to smooth out the overhang regions (0 - disable smoothing)
    int hops = 0;
    /// mesh transform
    const AffineXf3f* xf = nullptr;
};

/// \brief Find face regions that might create overhangs
/// \param mesh - source mesh
/// \param settings - parameters
/// \return face regions
MRMESH_API std::vector<FaceBitSet> findOverhangs( const Mesh& mesh, const FindOverhangsSettings& settings );

} // namespace MR