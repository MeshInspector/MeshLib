#pragma once

#include "MRMeshFwd.h"
#include <optional>

namespace MR
{

struct FaceDistancesSettings
{
    enum class OutputFaceValues
    {
        Distances, ///< each face will get its distance from start in the result
        SeqOrder   ///< each face will get its sequential order (1,2,...) from start in the result
    };
    OutputFaceValues out = OutputFaceValues::Distances;

    /// optional output of the maximal distance to the most distant face
    float * maxDist = nullptr;
    
    /// for progress reporting and cancellation
    ProgressCallback progress;
};

/// computes and returns the distance of traveling from one of start faces to all other reachable faces on the mesh;
/// all unreachable faces will get FLT_MAX value;
/// \param starts all start faces will get value 0 in the result;
/// \param metric metric(e) says the distance of traveling from left(e) to right(e)
[[nodiscard]] MRMESH_API std::optional<FaceScalars> calcFaceDistances( const MeshTopology & topology, const EdgeMetric & metric, const FaceBitSet & starts,
    const FaceDistancesSettings & settings = {} );

} // namespace MR
