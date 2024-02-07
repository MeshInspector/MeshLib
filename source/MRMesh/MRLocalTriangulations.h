#pragma once

#include "MRId.h"
#include "MRVector.h"
#include "MRBuffer.h"
#include <array>
#include <cstdint>
#include <optional>

namespace MR
{

/// describes one fan of triangles around a point excluding the point
struct FanRecord
{
    /// first border edge (invalid if the center point is not on the boundary);
    /// triangle associated with this point is absent
    VertId border;

    /// the position of first neigbor in LocalTriangulations::neighbours
    std::uint32_t firstNei;

    FanRecord( VertId b = {}, std::uint32_t fn = 0 ) : border( b ), firstNei( fn ) {}
    FanRecord( NoInit ) : border( noInit ) {}
};

/// describes one fan of triangles around a point including the point
struct FanRecordWithCenter : FanRecord
{
    /// center point in the fan
    VertId center;

    FanRecordWithCenter( VertId c = {}, VertId b = {}, std::uint32_t fn = 0 ) : FanRecord( b, fn ), center( c ) {}
    FanRecordWithCenter( NoInit ) : FanRecord( noInit ), center( noInit ) {}
};

/// describes a number of local triangulations of some points (e.g. assigned to a thread)
struct SomeLocalTriangulations
{
    std::vector<VertId> neighbors;
    std::vector<FanRecordWithCenter> fanRecords;
    VertId maxCenterId; //in fanRecords
};

/// triangulations for all points, with easy access by VertId
struct AllLocalTriangulations
{
    Buffer<VertId> neighbors;
    Vector<FanRecord, VertId> fanRecords;
};

/// converts a set of SomeLocalTriangulations containing local triangulations of all points arbitrary distributed among them
/// into one AllLocalTriangulations with records for all points
[[nodiscard]] MRMESH_API std::optional<AllLocalTriangulations> uniteLocalTriangulations( const std::vector<SomeLocalTriangulations> & in, const ProgressCallback & progress = {} );

/// orient neighbors around each point so they will be in clockwise order if look from the top of target normal
MRMESH_API void orientLocalTriangulations( AllLocalTriangulations & triangs, const VertCoords & coords, const VertNormals & normals );

/// TrianglesRepetitions[0] contains the number of unoriented triangles that appear in one local triangulation only
/// TrianglesRepetitions[1] contains the number of unoriented triangles that appear in exactly two local triangulations
/// TrianglesRepetitions[2] contains the number of unoriented triangles that appear in three local triangulations
using VotesForTriangles = std::array<int, 3>;

/// computes statistics about the number of triangle repetitions in local triangulations
[[nodiscard]] MRMESH_API VotesForTriangles computeTrianglesRepetitions( const AllLocalTriangulations & triangs );

/// from local triangulations returns all unoriented with given number of repetitions each in [0,2]
[[nodiscard]] MRMESH_API std::vector<UnorientedTriangle> findRepeatedTriangles( const AllLocalTriangulations & triangs, int repetitions );

} //namespace MR
