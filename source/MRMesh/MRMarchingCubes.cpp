#include "MRMarchingCubes.h"
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRIsNaN.h"
#include "MRMesh.h"
#include "MRVolumeIndexer.h"
#include "MRLine3.h"
#include "MRMeshBuilder.h"
#include "MRVDBFloatGrid.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MROpenvdb.h"
#include <thread>

namespace MR
{

namespace MarchingCubesHelper
{

enum class NeighborDir
{
    X, Y, Z, Count
};

// point between two neighbor voxels
struct SeparationPoint
{
    Vector3f position; // coordinate
    VertId vid; // any valid VertId is ok
    // each SeparationPointMap element has three SeparationPoint, it is not guaranteed that all three are valid (at least one is)
    // so there are some points present in map that are not valid
    explicit operator bool() const
    {
        return vid.valid();
    }
};

using SeparationPointSet = std::array<SeparationPoint, size_t( NeighborDir::Count )>;
using SeparationPointMap = ParallelHashMap<size_t, SeparationPointSet>;
template <size_t N> using ItersArray = std::array<SeparationPointMap::const_iterator, N>;

// lookup table from
// http://paulbourke.net/geometry/polygonise/
using EdgeDirIndex = std::pair<int, NeighborDir>;
constexpr std::array<EdgeDirIndex, 12> cEdgeIndicesMap = {
   EdgeDirIndex{0,NeighborDir::X},
   EdgeDirIndex{1,NeighborDir::Y},
   EdgeDirIndex{2,NeighborDir::X},
   EdgeDirIndex{0,NeighborDir::Y},

   EdgeDirIndex{4,NeighborDir::X},
   EdgeDirIndex{5,NeighborDir::Y},
   EdgeDirIndex{6,NeighborDir::X},
   EdgeDirIndex{4,NeighborDir::Y},

   EdgeDirIndex{0,NeighborDir::Z},
   EdgeDirIndex{1,NeighborDir::Z},
   EdgeDirIndex{3,NeighborDir::Z},
   EdgeDirIndex{2,NeighborDir::Z}
};

const std::array<Vector3i, 8> cVoxelNeighbors{
    Vector3i{0,0,0},
    Vector3i{1,0,0},
    Vector3i{0,1,0},
    Vector3i{1,1,0},
    Vector3i{0,0,1},
    Vector3i{1,0,1},
    Vector3i{0,1,1},
    Vector3i{1,1,1}
};
constexpr std::array<int, 8> cMapNeighborsShift{ 0,1,3,2,4,5,7,6 };

constexpr std::array<uint8_t, 8> cMapNeighbors
{
    1 << cMapNeighborsShift[0],
    1 << cMapNeighborsShift[1],
    1 << cMapNeighborsShift[2],
    1 << cMapNeighborsShift[3],
    1 << cMapNeighborsShift[4],
    1 << cMapNeighborsShift[5],
    1 << cMapNeighborsShift[6],
    1 << cMapNeighborsShift[7]
};

using TriangulationPlan = std::vector<int>;
const std::array<TriangulationPlan, 256> cTriangleTable = {
TriangulationPlan{},
TriangulationPlan{0, 8, 3},
TriangulationPlan{0, 1, 9},
TriangulationPlan{1, 8, 3, 9, 8, 1},
TriangulationPlan{1, 2, 10},
TriangulationPlan{0, 8, 3, 1, 2, 10},
TriangulationPlan{9, 2, 10, 0, 2, 9},
TriangulationPlan{2, 8, 3, 2, 10, 8, 10, 9, 8},
TriangulationPlan{3, 11, 2},
TriangulationPlan{0, 11, 2, 8, 11, 0},
TriangulationPlan{1, 9, 0, 2, 3, 11},
TriangulationPlan{1, 11, 2, 1, 9, 11, 9, 8, 11},
TriangulationPlan{3, 10, 1, 11, 10, 3},
TriangulationPlan{0, 10, 1, 0, 8, 10, 8, 11, 10},
TriangulationPlan{3, 9, 0, 3, 11, 9, 11, 10, 9},
TriangulationPlan{9, 8, 10, 10, 8, 11},
TriangulationPlan{4, 7, 8},
TriangulationPlan{4, 3, 0, 7, 3, 4},
TriangulationPlan{0, 1, 9, 8, 4, 7},
TriangulationPlan{4, 1, 9, 4, 7, 1, 7, 3, 1},
TriangulationPlan{1, 2, 10, 8, 4, 7},
TriangulationPlan{3, 4, 7, 3, 0, 4, 1, 2, 10},
TriangulationPlan{9, 2, 10, 9, 0, 2, 8, 4, 7},
TriangulationPlan{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4},
TriangulationPlan{8, 4, 7, 3, 11, 2},
TriangulationPlan{11, 4, 7, 11, 2, 4, 2, 0, 4},
TriangulationPlan{9, 0, 1, 8, 4, 7, 2, 3, 11},
TriangulationPlan{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1},
TriangulationPlan{3, 10, 1, 3, 11, 10, 7, 8, 4},
TriangulationPlan{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4},
TriangulationPlan{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3},
TriangulationPlan{4, 7, 11, 4, 11, 9, 9, 11, 10},
TriangulationPlan{9, 5, 4},
TriangulationPlan{9, 5, 4, 0, 8, 3},
TriangulationPlan{0, 5, 4, 1, 5, 0},
TriangulationPlan{8, 5, 4, 8, 3, 5, 3, 1, 5},
TriangulationPlan{1, 2, 10, 9, 5, 4},
TriangulationPlan{3, 0, 8, 1, 2, 10, 4, 9, 5},
TriangulationPlan{5, 2, 10, 5, 4, 2, 4, 0, 2},
TriangulationPlan{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8},
TriangulationPlan{9, 5, 4, 2, 3, 11},
TriangulationPlan{0, 11, 2, 0, 8, 11, 4, 9, 5},
TriangulationPlan{0, 5, 4, 0, 1, 5, 2, 3, 11},
TriangulationPlan{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5},
TriangulationPlan{10, 3, 11, 10, 1, 3, 9, 5, 4},
TriangulationPlan{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10},
TriangulationPlan{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3},
TriangulationPlan{5, 4, 8, 5, 8, 10, 10, 8, 11},
TriangulationPlan{9, 7, 8, 5, 7, 9},
TriangulationPlan{9, 3, 0, 9, 5, 3, 5, 7, 3},
TriangulationPlan{0, 7, 8, 0, 1, 7, 1, 5, 7},
TriangulationPlan{1, 5, 3, 3, 5, 7},
TriangulationPlan{9, 7, 8, 9, 5, 7, 10, 1, 2},
TriangulationPlan{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3},
TriangulationPlan{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2},
TriangulationPlan{2, 10, 5, 2, 5, 3, 3, 5, 7},
TriangulationPlan{7, 9, 5, 7, 8, 9, 3, 11, 2},
TriangulationPlan{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11},
TriangulationPlan{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7},
TriangulationPlan{11, 2, 1, 11, 1, 7, 7, 1, 5},
TriangulationPlan{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11},
TriangulationPlan{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0},
TriangulationPlan{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0},
TriangulationPlan{11, 10, 5, 7, 11, 5},
TriangulationPlan{10, 6, 5},
TriangulationPlan{0, 8, 3, 5, 10, 6},
TriangulationPlan{9, 0, 1, 5, 10, 6},
TriangulationPlan{1, 8, 3, 1, 9, 8, 5, 10, 6},
TriangulationPlan{1, 6, 5, 2, 6, 1},
TriangulationPlan{1, 6, 5, 1, 2, 6, 3, 0, 8},
TriangulationPlan{9, 6, 5, 9, 0, 6, 0, 2, 6},
TriangulationPlan{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8},
TriangulationPlan{2, 3, 11, 10, 6, 5},
TriangulationPlan{11, 0, 8, 11, 2, 0, 10, 6, 5},
TriangulationPlan{0, 1, 9, 2, 3, 11, 5, 10, 6},
TriangulationPlan{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11},
TriangulationPlan{6, 3, 11, 6, 5, 3, 5, 1, 3},
TriangulationPlan{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6},
TriangulationPlan{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9},
TriangulationPlan{6, 5, 9, 6, 9, 11, 11, 9, 8},
TriangulationPlan{5, 10, 6, 4, 7, 8},
TriangulationPlan{4, 3, 0, 4, 7, 3, 6, 5, 10},
TriangulationPlan{1, 9, 0, 5, 10, 6, 8, 4, 7},
TriangulationPlan{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4},
TriangulationPlan{6, 1, 2, 6, 5, 1, 4, 7, 8},
TriangulationPlan{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7},
TriangulationPlan{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6},
TriangulationPlan{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9},
TriangulationPlan{3, 11, 2, 7, 8, 4, 10, 6, 5},
TriangulationPlan{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11},
TriangulationPlan{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6},
TriangulationPlan{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6},
TriangulationPlan{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6},
TriangulationPlan{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11},
TriangulationPlan{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7},
TriangulationPlan{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9},
TriangulationPlan{10, 4, 9, 6, 4, 10},
TriangulationPlan{4, 10, 6, 4, 9, 10, 0, 8, 3},
TriangulationPlan{10, 0, 1, 10, 6, 0, 6, 4, 0},
TriangulationPlan{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10},
TriangulationPlan{1, 4, 9, 1, 2, 4, 2, 6, 4},
TriangulationPlan{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4},
TriangulationPlan{0, 2, 4, 4, 2, 6},
TriangulationPlan{8, 3, 2, 8, 2, 4, 4, 2, 6},
TriangulationPlan{10, 4, 9, 10, 6, 4, 11, 2, 3},
TriangulationPlan{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6},
TriangulationPlan{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10},
TriangulationPlan{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1},
TriangulationPlan{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3},
TriangulationPlan{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1},
TriangulationPlan{3, 11, 6, 3, 6, 0, 0, 6, 4},
TriangulationPlan{6, 4, 8, 11, 6, 8},
TriangulationPlan{7, 10, 6, 7, 8, 10, 8, 9, 10},
TriangulationPlan{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10},
TriangulationPlan{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0},
TriangulationPlan{10, 6, 7, 10, 7, 1, 1, 7, 3},
TriangulationPlan{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7},
TriangulationPlan{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9},
TriangulationPlan{7, 8, 0, 7, 0, 6, 6, 0, 2},
TriangulationPlan{7, 3, 2, 6, 7, 2},
TriangulationPlan{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7},
TriangulationPlan{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7},
TriangulationPlan{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11},
TriangulationPlan{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1},
TriangulationPlan{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6},
TriangulationPlan{0, 9, 1, 11, 6, 7},
TriangulationPlan{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0},
TriangulationPlan{7, 11, 6},
TriangulationPlan{7, 6, 11},
TriangulationPlan{3, 0, 8, 11, 7, 6},
TriangulationPlan{0, 1, 9, 11, 7, 6},
TriangulationPlan{8, 1, 9, 8, 3, 1, 11, 7, 6},
TriangulationPlan{10, 1, 2, 6, 11, 7},
TriangulationPlan{1, 2, 10, 3, 0, 8, 6, 11, 7},
TriangulationPlan{2, 9, 0, 2, 10, 9, 6, 11, 7},
TriangulationPlan{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8},
TriangulationPlan{7, 2, 3, 6, 2, 7},
TriangulationPlan{7, 0, 8, 7, 6, 0, 6, 2, 0},
TriangulationPlan{2, 7, 6, 2, 3, 7, 0, 1, 9},
TriangulationPlan{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6},
TriangulationPlan{10, 7, 6, 10, 1, 7, 1, 3, 7},
TriangulationPlan{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8},
TriangulationPlan{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7},
TriangulationPlan{7, 6, 10, 7, 10, 8, 8, 10, 9},
TriangulationPlan{6, 8, 4, 11, 8, 6},
TriangulationPlan{3, 6, 11, 3, 0, 6, 0, 4, 6},
TriangulationPlan{8, 6, 11, 8, 4, 6, 9, 0, 1},
TriangulationPlan{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6},
TriangulationPlan{6, 8, 4, 6, 11, 8, 2, 10, 1},
TriangulationPlan{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6},
TriangulationPlan{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9},
TriangulationPlan{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3},
TriangulationPlan{8, 2, 3, 8, 4, 2, 4, 6, 2},
TriangulationPlan{0, 4, 2, 4, 6, 2},
TriangulationPlan{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8},
TriangulationPlan{1, 9, 4, 1, 4, 2, 2, 4, 6},
TriangulationPlan{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1},
TriangulationPlan{10, 1, 0, 10, 0, 6, 6, 0, 4},
TriangulationPlan{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3},
TriangulationPlan{10, 9, 4, 6, 10, 4},
TriangulationPlan{4, 9, 5, 7, 6, 11},
TriangulationPlan{0, 8, 3, 4, 9, 5, 11, 7, 6},
TriangulationPlan{5, 0, 1, 5, 4, 0, 7, 6, 11},
TriangulationPlan{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5},
TriangulationPlan{9, 5, 4, 10, 1, 2, 7, 6, 11},
TriangulationPlan{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5},
TriangulationPlan{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2},
TriangulationPlan{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6},
TriangulationPlan{7, 2, 3, 7, 6, 2, 5, 4, 9},
TriangulationPlan{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7},
TriangulationPlan{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0},
TriangulationPlan{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8},
TriangulationPlan{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7},
TriangulationPlan{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4},
TriangulationPlan{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10},
TriangulationPlan{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10},
TriangulationPlan{6, 9, 5, 6, 11, 9, 11, 8, 9},
TriangulationPlan{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5},
TriangulationPlan{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11},
TriangulationPlan{6, 11, 3, 6, 3, 5, 5, 3, 1},
TriangulationPlan{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6},
TriangulationPlan{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10},
TriangulationPlan{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5},
TriangulationPlan{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3},
TriangulationPlan{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2},
TriangulationPlan{9, 5, 6, 9, 6, 0, 0, 6, 2},
TriangulationPlan{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8},
TriangulationPlan{1, 5, 6, 2, 1, 6},
TriangulationPlan{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6},
TriangulationPlan{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0},
TriangulationPlan{0, 3, 8, 5, 6, 10},
TriangulationPlan{10, 5, 6},
TriangulationPlan{11, 5, 10, 7, 5, 11},
TriangulationPlan{11, 5, 10, 11, 7, 5, 8, 3, 0},
TriangulationPlan{5, 11, 7, 5, 10, 11, 1, 9, 0},
TriangulationPlan{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1},
TriangulationPlan{11, 1, 2, 11, 7, 1, 7, 5, 1},
TriangulationPlan{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11},
TriangulationPlan{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7},
TriangulationPlan{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2},
TriangulationPlan{2, 5, 10, 2, 3, 5, 3, 7, 5},
TriangulationPlan{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5},
TriangulationPlan{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2},
TriangulationPlan{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2},
TriangulationPlan{1, 3, 5, 3, 7, 5},
TriangulationPlan{0, 8, 7, 0, 7, 1, 1, 7, 5},
TriangulationPlan{9, 0, 3, 9, 3, 5, 5, 3, 7},
TriangulationPlan{9, 8, 7, 5, 9, 7},
TriangulationPlan{5, 8, 4, 5, 10, 8, 10, 11, 8},
TriangulationPlan{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0},
TriangulationPlan{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5},
TriangulationPlan{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4},
TriangulationPlan{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8},
TriangulationPlan{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11},
TriangulationPlan{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5},
TriangulationPlan{9, 4, 5, 2, 11, 3},
TriangulationPlan{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4},
TriangulationPlan{5, 10, 2, 5, 2, 4, 4, 2, 0},
TriangulationPlan{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9},
TriangulationPlan{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2},
TriangulationPlan{8, 4, 5, 8, 5, 3, 3, 5, 1},
TriangulationPlan{0, 4, 5, 1, 0, 5},
TriangulationPlan{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5},
TriangulationPlan{9, 4, 5},
TriangulationPlan{4, 11, 7, 4, 9, 11, 9, 10, 11},
TriangulationPlan{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11},
TriangulationPlan{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11},
TriangulationPlan{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4},
TriangulationPlan{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2},
TriangulationPlan{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3},
TriangulationPlan{11, 7, 4, 11, 4, 2, 2, 4, 0},
TriangulationPlan{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4},
TriangulationPlan{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9},
TriangulationPlan{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7},
TriangulationPlan{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10},
TriangulationPlan{1, 10, 2, 8, 7, 4},
TriangulationPlan{4, 9, 1, 4, 1, 7, 7, 1, 3},
TriangulationPlan{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1},
TriangulationPlan{4, 0, 3, 7, 4, 3},
TriangulationPlan{4, 8, 7},
TriangulationPlan{9, 10, 8, 10, 11, 8},
TriangulationPlan{3, 0, 9, 3, 9, 11, 11, 9, 10},
TriangulationPlan{0, 1, 10, 0, 10, 8, 8, 10, 11},
TriangulationPlan{3, 1, 10, 11, 3, 10},
TriangulationPlan{1, 2, 11, 1, 11, 9, 9, 11, 8},
TriangulationPlan{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9},
TriangulationPlan{0, 2, 11, 8, 0, 11},
TriangulationPlan{3, 2, 11},
TriangulationPlan{2, 3, 8, 2, 8, 10, 10, 8, 9},
TriangulationPlan{9, 10, 2, 0, 9, 2},
TriangulationPlan{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8},
TriangulationPlan{1, 10, 2},
TriangulationPlan{1, 3, 8, 9, 1, 8},
TriangulationPlan{0, 9, 1},
TriangulationPlan{0, 3, 8},
TriangulationPlan{}
};

const std::array<OutEdge, size_t( NeighborDir::Count )> cOutEdgeMap { OutEdge::PlusX, OutEdge::PlusY, OutEdge::PlusZ };

// each iterator has info about separation point on plus directions of it base
// mode: 0 - (0,0,0) voxel, +x, +y, +z possible separation points
// mode: 1 - (1,0,0) voxel,     +y, +z possible separation points
// mode: 2 - (0,1,0) voxel, +x,     +z possible separation points
// mode: 3 - (1,1,0) voxel,         +z possible separation points
// mode: 4 - (0,0,1) voxel, +x, +y     possible separation points
// mode: 5 - (1,0,1) voxel,     +y     possible separation points
// mode: 6 - (0,1,1) voxel, +x         possible separation points
// 
// function returns true if given voxelsConfig requires separation points in given mode
bool cNeedIteratorMode( int mode, uint8_t voxelsConfig )
{
    if ( mode == 0 )
    {
        auto base = ( voxelsConfig & cMapNeighbors[0] );
        if ( base != ( voxelsConfig & cMapNeighbors[1] ) )
            return true;
        else if ( base != ( voxelsConfig & cMapNeighbors[2] ) )
            return true;
        else if ( base != ( voxelsConfig & cMapNeighbors[4] ) )
            return true;
        else
            return false;
    }
    else if ( mode == 1 )
    {
        auto base = ( voxelsConfig & cMapNeighbors[1] );
        if ( base != ( voxelsConfig & cMapNeighbors[3] ) )
            return true;
        else if ( base != ( voxelsConfig & cMapNeighbors[5] ) )
            return true;
        else
            return false;
    }
    else if ( mode == 2 )
    {
        auto base = ( voxelsConfig & cMapNeighbors[2] );
        if ( base != ( voxelsConfig & cMapNeighbors[3] ) )
            return true;
        else if ( base != ( voxelsConfig & cMapNeighbors[6] ) )
            return true;
        else
            return false;
    }
    else if ( mode == 3 )
    {
        if ( ( voxelsConfig & cMapNeighbors[3] ) != ( voxelsConfig & cMapNeighbors[7] ) )
            return true;
        return false;
    }
    else if ( mode == 4 )
    {
        auto base = ( voxelsConfig & cMapNeighbors[4] );
        if ( base != ( voxelsConfig & cMapNeighbors[5] ) )
            return true;
        else if ( base != ( voxelsConfig & cMapNeighbors[6] ) )
            return true;
        else
            return false;
    }
    else if ( mode == 5 )
    {
        if ( ( voxelsConfig & cMapNeighbors[5] ) != ( voxelsConfig & cMapNeighbors[7] ) )
            return true;
        return false;
    }
    else if ( mode == 6 )
    {
        if ( ( voxelsConfig & cMapNeighbors[6] ) != ( voxelsConfig & cMapNeighbors[7] ) )
            return true;
        return false;
    }
    return false;
}

// mode: 0 - (0,0,0) voxel, +x, +y, +z possible separation points
// mode: 1 - (1,0,0) voxel,     +y, +z possible separation points
// mode: 2 - (0,1,0) voxel, +x,     +z possible separation points
// mode: 3 - (1,1,0) voxel,         +z possible separation points
// mode: 4 - (0,0,1) voxel, +x, +y     possible separation points
// mode: 5 - (1,0,1) voxel,     +y     possible separation points
// mode: 6 - (0,1,1) voxel, +x         possible separation points
// 
// function returns true if given set has at least one valid SeparationPoint
bool checkSetValid( const SeparationPointSet& set, int mode )
{
    switch ( mode )
    {
    case 0: // base voxel
        return true;
    case 1: // x + 1 voxel
        return set[int( NeighborDir::Y )] || set[int( NeighborDir::Z )];
    case 2: // y + 1 voxel
        return set[int( NeighborDir::X )] || set[int( NeighborDir::Z )];
    case 3: // x + 1, y + 1 voxel
        return bool( set[int( NeighborDir::Z )] );
    case 4: // z + 1 voxel
        return set[int( NeighborDir::X )] || set[int( NeighborDir::Y )];
    case 5: // x + 1, z + 1 voxel
        return bool( set[int( NeighborDir::Y )] );
    case 6: // y + 1, z + 1 voxel
        return bool( set[int( NeighborDir::X )] );
    default:
        return false;
    }
}

}

using namespace MarchingCubesHelper;
using ConstAccessor = openvdb::FloatGrid::ConstAccessor;
using VdbCoord = openvdb::Coord;

inline Vector3f voxelPositionerLinearInline( const Vector3f& pos0, const Vector3f& pos1, float v0, float v1, float iso )
{
    const auto ratio = std::clamp( ( iso - v0 ) / ( v1 - v0 ), 0.0f, 1.0f );
    return ( 1.0f - ratio ) * pos0 + ratio * pos1;
}

Vector3f voxelPositionerLinear( const Vector3f& pos0, const Vector3f& pos1, float v0, float v1, float iso )
{
    return voxelPositionerLinearInline( pos0, pos1, v0, v1, iso );
}

template <bool UseDefaultVoxelPointPositioner>
bool findSeparationPoint( SeparationPoint& sp, const VdbVolume& volume, const ConstAccessor& acc,
                          const openvdb::Coord& coord, const Vector3i& basePos, float valueB, NeighborDir dir,
                          const MarchingCubesParams& params )
{
    if ( basePos[int( dir )] + 1 >= volume.dims[int( dir )] )
        return false;
    auto nextCoord = coord;
    nextCoord[int( dir )] += 1;
    float valueD = acc.getValue( nextCoord );// volume.data[nextId];

    bool bLower = valueB < params.iso;
    bool dLower = valueD < params.iso;
    if ( bLower == dLower )
        return false;

    Vector3f coordF = Vector3f( float( coord.x() ), float( coord.y() ), float( coord.z() ) );
    Vector3f nextCoordF = Vector3f( float( nextCoord.x() ), float( nextCoord.y() ), float( nextCoord.z() ) );
    auto bPos = params.origin + mult( volume.voxelSize, coordF );
    auto dPos = params.origin + mult( volume.voxelSize, nextCoordF );
    if constexpr ( UseDefaultVoxelPointPositioner )
        sp.position = voxelPositionerLinearInline( bPos, dPos, valueB, valueD, params.iso );
    else
        sp.position = params.positioner( bPos, dPos, valueB, valueD, params.iso );
    return true;
}

template <typename NaNChecker, bool UseDefaultVoxelPointPositioner>
bool findSeparationPoint( SeparationPoint& sp, const SimpleVolume& volume, const VolumeIndexer& indexer, VoxelId base,
                          const Vector3i& basePos, NeighborDir dir, const MarchingCubesParams& params, NaNChecker&& nanChecker )
{
    auto nextPos = basePos;
    nextPos[int( dir )] += 1;
    if ( nextPos[int( dir )] >= volume.dims[int( dir )] )
        return false;

    float valueB = volume.data[base];
    float valueD = volume.data[indexer.getExistingNeighbor( base, cOutEdgeMap[int( dir )] ).get()];
    if ( nanChecker( valueB ) || nanChecker( valueD ) )
        return false;

    bool bLower = valueB < params.iso;
    bool dLower = valueD < params.iso;
    if ( bLower == dLower )
        return false;

    Vector3f coordF = Vector3f( basePos ) + Vector3f::diagonal( 0.5f );
    Vector3f nextCoordF = Vector3f( nextPos ) + Vector3f::diagonal( 0.5f );
    auto bPos = params.origin + mult( volume.voxelSize, coordF );
    auto dPos = params.origin + mult( volume.voxelSize, nextCoordF );
    if constexpr ( UseDefaultVoxelPointPositioner )
        sp.position = voxelPositionerLinearInline( bPos, dPos, valueB, valueD, params.iso );
    else
        sp.position = params.positioner( bPos, dPos, valueB, valueD, params.iso );
    return true;
}

template <typename NaNChecker, bool UseDefaultVoxelPointPositioner>
bool findSeparationPoint( SeparationPoint& sp, const FunctionVolume& volume, const Vector3i& basePos, NeighborDir dir,
                          const MarchingCubesParams& params, NaNChecker&& nanChecker )
{
    auto nextPos = basePos;
    nextPos[int( dir )] += 1;
    if ( nextPos[int( dir )] >= volume.dims[int( dir )] )
        return false;

    float valueB = volume.data( basePos );
    float valueD = volume.data( nextPos );
    if ( nanChecker( valueB ) || nanChecker( valueD ) )
        return false;

    bool bLower = valueB < params.iso;
    bool dLower = valueD < params.iso;
    if ( bLower == dLower )
        return false;

    Vector3f coordF = Vector3f( basePos ) + Vector3f::diagonal( 0.5f );
    Vector3f nextCoordF = Vector3f( nextPos ) + Vector3f::diagonal( 0.5f );
    auto bPos = params.origin + mult( volume.voxelSize, coordF );
    auto dPos = params.origin + mult( volume.voxelSize, nextCoordF );
    if constexpr ( UseDefaultVoxelPointPositioner )
        sp.position = voxelPositionerLinearInline( bPos, dPos, valueB, valueD, params.iso );
    else
        sp.position = params.positioner( bPos, dPos, valueB, valueD, params.iso );
    return true;
}

template<typename V> auto accessorCtor( const V& v );

template<> auto accessorCtor<SimpleVolume>( const SimpleVolume& ) { return ( void* )nullptr; }

template<> auto accessorCtor<VdbVolume>( const VdbVolume& v ) { return v.data->getConstAccessor(); }

template<> auto accessorCtor<FunctionVolume>( const FunctionVolume& ) { return (void*)nullptr; }

template<typename V, typename NaNChecker, bool UseDefaultVoxelPointPositioner>
Expected<Mesh, std::string> volumeToMesh( const V& volume, const MarchingCubesParams& params, NaNChecker&& nanChecker )
{
    if constexpr ( std::is_same_v<V, VdbVolume> )
    {
        if ( !volume.data )
            return unexpected( "No volume data." );
    }
    else if constexpr ( std::is_same_v<V, FunctionVolume> )
    {
        if ( !volume.data )
            return unexpected( "Getter function is not specified." );
    }

    Mesh result;
    if ( params.iso <= volume.min || params.iso >= volume.max ||
        volume.dims.x <= 0 || volume.dims.y <= 0 || volume.dims.z <= 0 )
        return result;

    MR_TIMER

    VdbCoord minCoord;
    if constexpr ( std::is_same_v<V, VdbVolume> )
        minCoord = volume.data->evalActiveVoxelBoundingBox().min();

    VolumeIndexer indexer( volume.dims );

    std::atomic<bool> keepGoing{ true };
    auto mainThreadId = std::this_thread::get_id();
    int lastSubMap = -1;

    size_t threadCount = tbb::global_control::parameter( tbb::global_control::max_allowed_parallelism );
    if ( threadCount == 0 )
        threadCount = std::thread::hardware_concurrency();
    if ( threadCount == 0 )
        threadCount = 1;

    const auto blockCount = threadCount;
    const auto blockSize = (size_t)std::ceil( (float)indexer.size() / blockCount );
    assert( indexer.size() <= blockSize * blockCount );

    std::vector<SeparationPointMap> hmaps( blockCount );
    auto hmap = [&] ( size_t index ) -> SeparationPointMap&
    {
        return hmaps[index / blockSize];
    };

    // find all separate points
    // fill map in parallel
    struct VertsNumeration
    {
        // explicit ctor to fix clang build with `vec.emplace_back( ind, 0 )`
        VertsNumeration( size_t ind, size_t num ) :initIndex{ ind }, numVerts{ num }{}
        size_t initIndex{ 0 };
        size_t numVerts{ 0 };
    };
    using PerThreadVertNumeration = std::vector<VertsNumeration>;
    tbb::enumerable_thread_specific<PerThreadVertNumeration> perThreadVertNumeration;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, blockCount, 1 ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        const auto blockIndex = range.begin();

        // vdb version cache
        [[maybe_unused]] auto acc = accessorCtor( volume );
        [[maybe_unused]] VdbCoord baseCoord;
        [[maybe_unused]] float baseValue{ 0.0f };

        if ( std::this_thread::get_id() == mainThreadId && lastSubMap == -1 )
            lastSubMap = int( blockIndex );
        const bool runCallback = params.cb && std::this_thread::get_id() == mainThreadId && lastSubMap == blockIndex;

        const auto begin = blockIndex * blockSize;
        const auto end = std::min( ( blockIndex + 1 ) * blockSize, indexer.size() );

        auto& localNumeration = perThreadVertNumeration.local();
        localNumeration.emplace_back( begin, 0 );
        auto& thisRangeNumeration = localNumeration.back().numVerts;

        for ( size_t i = begin; i < end; ++i )
        {
            if ( params.cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            SeparationPointSet set;
            bool atLeastOneOk = false;
            auto basePos = indexer.toPos( VoxelId( i ) );
            if constexpr ( std::is_same_v<V, VdbVolume> )
            {
                baseCoord = openvdb::Coord{ basePos.x + minCoord.x(), basePos.y + minCoord.y(), basePos.z + minCoord.z() };
                baseValue = acc.getValue( baseCoord );
            }
            for ( int n = int( NeighborDir::X ); n < int( NeighborDir::Count ); ++n )
            {
                bool ok = false;
                if constexpr ( std::is_same_v<V, VdbVolume> )
                    ok = findSeparationPoint<UseDefaultVoxelPointPositioner>( set[n], volume, acc, baseCoord, basePos, baseValue, NeighborDir( n ), params );
                else if constexpr ( std::is_same_v<V, SimpleVolume> )
                    ok = findSeparationPoint<NaNChecker, UseDefaultVoxelPointPositioner>( set[n], volume, indexer, VoxelId( i ), basePos, NeighborDir( n ), params, std::forward<NaNChecker>( nanChecker ) );
                else if constexpr ( std::is_same_v<V, FunctionVolume> )
                    ok = findSeparationPoint<NaNChecker, UseDefaultVoxelPointPositioner>( set[n], volume, basePos, NeighborDir( n ), params, std::forward<NaNChecker>( nanChecker ) );
                else
                    static_assert( !sizeof( V ), "Unsupported voxel volume type." );

                if ( ok )
                {
                    set[n].vid = VertId( thisRangeNumeration++ );
                    atLeastOneOk = true;
                }
            }

            if ( runCallback && ( i - begin ) % 1024 == 0 )
                if ( !params.cb( 0.3f * float( i - begin ) / float( end - begin ) ) )
                    keepGoing.store( false, std::memory_order_relaxed );

            if ( !atLeastOneOk )
                continue;

            hmap( i ).insert( { i, set } );
        }
    } );

    if ( params.cb && !keepGoing )
        return unexpectedOperationCanceled();

    // organize vert numeration
    std::vector<VertsNumeration> resultVertNumeration;
    size_t totalVertices = 0;
    for ( auto& perThreadNum : perThreadVertNumeration )
    {
        for ( auto & obj : perThreadNum )
        {
            totalVertices += obj.numVerts;
            if ( obj.numVerts > 0 )
                resultVertNumeration.push_back( std::move( obj ) );
        }
        perThreadNum.clear();
    }
    if ( totalVertices > params.maxVertices )
        return unexpected( "Vertices number limit exceeded." );

    // sort by voxel index
    std::sort( resultVertNumeration.begin(), resultVertNumeration.end(), [] ( const auto& l, const auto& r )
    {
        return l.initIndex < r.initIndex;
    } );

    auto getVertIndexShiftForVoxelId = [&] ( size_t ind )
    {
        size_t shift = 0;
        for ( int i = 1; i < resultVertNumeration.size(); ++i )
        {
            if ( ind >= resultVertNumeration[i].initIndex )
                shift += resultVertNumeration[i - 1].numVerts;
        }
        return VertId( shift );
    };

    // update map with determined vert indices
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, hmaps.size(), 1 ),
    [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        for ( auto& [ind, set] : hmaps[range.begin()] )
        {
            auto vertShift = getVertIndexShiftForVoxelId( ind );
            for ( auto& sepPoint : set )
                if ( sepPoint )
                    sepPoint.vid += vertShift;
        }
    } );


    if ( params.cb && !params.cb( 0.5f ) )
        return unexpectedOperationCanceled();

    // triangulate by table
    struct TriangulationData
    {
        size_t initInd{ 0 }; // this is needed to have determined topology independent of threads number
        Triangulation t;
        Vector<VoxelId, FaceId> faceMap;
    };
    using PerThreadTriangulation = std::vector<TriangulationData>;
    auto subprogress2 = MR::subprogress( params.cb, 0.5f, 0.85f );
    std::atomic<size_t> voxelsDone{0};

    const std::array<size_t, 8> cVoxelNeighborsIndexAdd = 
    {
        0,
        1,
        size_t( indexer.dims().x ),
        size_t( indexer.dims().x ) + 1,
        indexer.sizeXY(),
        indexer.sizeXY() + 1,
        indexer.sizeXY() + size_t( indexer.dims().x ),
        indexer.sizeXY() + size_t( indexer.dims().x ) + 1
    };

    tbb::enumerable_thread_specific<PerThreadTriangulation> triangulationPerThread;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, indexer.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        // setup local triangulation
        auto& localTriangulatoinData = triangulationPerThread.local();
        localTriangulatoinData.emplace_back();
        auto& thisTriData = localTriangulatoinData.back();
        thisTriData.initInd = range.begin();
        auto& t = thisTriData.t;
        auto& faceMap = thisTriData.faceMap;

        // vdb accessor
        [[maybe_unused]] auto acc = accessorCtor( volume );

        // cell data
        ItersArray<7> iters;
        std::array<bool, 7> iterStatus;
        unsigned char voxelConfiguration;
        for ( size_t ind = range.begin(); ind < range.end(); ++ind )
        {
            if ( subprogress2 && !keepGoing.load( std::memory_order_relaxed ) )
                break;
            Vector3i basePos = indexer.toPos( VoxelId( ind ) );
            if ( basePos.x + 1 >= volume.dims.x ||
                basePos.y + 1 >= volume.dims.y ||
                basePos.z + 1 >= volume.dims.z )
                continue;

            bool voxelValid = true;
            voxelConfiguration = 0;
            [[maybe_unused]] bool atLeastOneNan = false;
            for ( int i = 0; i < cVoxelNeighbors.size(); ++i )
            {
                auto pos = basePos + cVoxelNeighbors[i];
                float value{ 0.0f };
                if constexpr ( std::is_same_v<V, VdbVolume> )
                {
                    value = acc.getValue( { pos.x + minCoord.x(),pos.y + minCoord.y(),pos.z + minCoord.z() } );
                }
                else if constexpr ( std::is_same_v<V, SimpleVolume> || std::is_same_v<V, FunctionVolume> )
                {
                    if constexpr ( std::is_same_v<V, SimpleVolume> )
                        value = volume.data[ind + cVoxelNeighborsIndexAdd[i]];
                    else
                        value = volume.data( pos );
                    // find non nan neighbor
                    constexpr std::array<uint8_t, 7> cNeighborsOrder{
                        0b001,
                        0b010,
                        0b100,
                        0b011,
                        0b101,
                        0b110,
                        0b111
                    };
                    int neighIndex = 0;
                    // iterates over nan neighbors to find consistent value
                    while ( nanChecker( value ) && neighIndex < 7 )
                    {
                        auto neighPos = pos;
                        for ( int posCoord = 0; posCoord < 3; ++posCoord )
                        {
                            int sign = 1;
                            if ( cVoxelNeighbors[i][posCoord] == 1 )
                                sign = -1;
                            neighPos[posCoord] += ( sign *
                                ( ( cNeighborsOrder[neighIndex] & ( 1 << posCoord ) ) >> posCoord ) );
                        }
                        if constexpr ( std::is_same_v<V, SimpleVolume> )
                            value = volume.data[indexer.toVoxelId( neighPos ).get()];
                        else
                            value = volume.data( neighPos );
                        ++neighIndex;
                    }
                    if ( nanChecker( value ) )
                    {
                        voxelValid = false;
                        break;
                    }
                    if ( !atLeastOneNan && neighIndex > 0 )
                        atLeastOneNan = true;
                }
                else
                {
                    static_assert( !sizeof( V ), "Unsupported voxel volume type." );
                }
                
                if ( value >= params.iso )
                    continue;
                voxelConfiguration |= cMapNeighbors[i];
            }
            if ( !voxelValid || voxelConfiguration == 0x00 || voxelConfiguration == 0xff )
                continue;

            voxelValid = false;
            for ( int i = 0; i < iters.size(); ++i )
            {
                if ( !cNeedIteratorMode( i, voxelConfiguration ) )
                {
                    iters[i] = {};
                    iterStatus[i] = false;
                    continue;
                }
                const auto index = ind + cVoxelNeighborsIndexAdd[i];
                iters[i] = hmap( index ).find( index );
                iterStatus[i] = ( iters[i] != hmap( index ).cend() ) && checkSetValid( iters[i]->second, i );
                if ( !voxelValid && iterStatus[i] )
                    voxelValid = true;
            }
            if constexpr ( std::is_same_v<V, SimpleVolume> || std::is_same_v<V, FunctionVolume> )
            {
                // ensure consistent nan voxel
                if ( atLeastOneNan && voxelValid )
                {
                    const auto& plan = cTriangleTable[voxelConfiguration];
                    for ( int i = 0; i < plan.size() && voxelValid; i += 3 )
                    {
                        const auto& [interIndex0, dir0] = cEdgeIndicesMap[plan[i]];
                        const auto& [interIndex1, dir1] = cEdgeIndicesMap[plan[i + 1]];
                        const auto& [interIndex2, dir2] = cEdgeIndicesMap[plan[i + 2]];
                        // `iterStatus` indicates that current voxel has valid point for desired triangulation
                        // as far as iter has 3 directions we use `dir` to validate (make sure that there is point in needed edge) desired direction
                        voxelValid = voxelValid && ( iterStatus[interIndex0] && iters[interIndex0]->second[int( dir0 )].vid );
                        voxelValid = voxelValid && ( iterStatus[interIndex1] && iters[interIndex1]->second[int( dir1 )].vid );
                        voxelValid = voxelValid && ( iterStatus[interIndex2] && iters[interIndex2]->second[int( dir2 )].vid );
                    }
                }
                if ( !voxelValid )
                    continue;
            }

            const auto& plan = cTriangleTable[voxelConfiguration];
            for ( int i = 0; i < plan.size(); i += 3 )
            {
                const auto& [interIndex0, dir0] = cEdgeIndicesMap[plan[i]];
                const auto& [interIndex1, dir1] = cEdgeIndicesMap[plan[i + 1]];
                const auto& [interIndex2, dir2] = cEdgeIndicesMap[plan[i + 2]];
                assert( iterStatus[interIndex0] && iters[interIndex0]->second[int( dir0 )].vid );
                assert( iterStatus[interIndex1] && iters[interIndex1]->second[int( dir1 )].vid );
                assert( iterStatus[interIndex2] && iters[interIndex2]->second[int( dir2 )].vid );

                if ( params.lessInside )
                    t.emplace_back( ThreeVertIds{
                    iters[interIndex0]->second[int( dir0 )].vid,
                    iters[interIndex2]->second[int( dir2 )].vid,
                    iters[interIndex1]->second[int( dir1 )].vid
                    } );
                else
                    t.emplace_back( ThreeVertIds{
                    iters[interIndex0]->second[int( dir0 )].vid,
                    iters[interIndex1]->second[int( dir1 )].vid,
                    iters[interIndex2]->second[int( dir2 )].vid
                    } );
                if ( params.outVoxelPerFaceMap )
                    faceMap.emplace_back( VoxelId{ ind } );
            }
        }

        if ( subprogress2 )
        {
            voxelsDone += range.size();
            if ( std::this_thread::get_id() == mainThreadId && !subprogress2( float( voxelsDone ) / float( indexer.size() ) ) )
                keepGoing.store( false, std::memory_order_relaxed );
        }
    } );

    if ( params.cb && !keepGoing )
        return unexpectedOperationCanceled();

    // organize per thread triangulation
    std::vector<TriangulationData> resTriangulatoinData;
    for ( auto& threadTriData : triangulationPerThread )
    {
        // remove empty
        threadTriData.erase( std::remove_if( threadTriData.begin(), threadTriData.end(),
            [] ( const auto& obj )
        {
            return obj.t.empty();
        } ), threadTriData.end() );
        if ( threadTriData.empty() )
            continue;
        // accum not empty
        resTriangulatoinData.insert( resTriangulatoinData.end(),
            std::make_move_iterator( threadTriData.begin() ), std::make_move_iterator( threadTriData.end() ) );
    }
    // sort by voxel index
    tbb::parallel_sort( resTriangulatoinData.begin(), resTriangulatoinData.end(), [] ( const auto& l, const auto& r ) { return l.initInd < r.initInd; } );

    // create result triangulation
    Triangulation resTriangulation;
    if ( params.outVoxelPerFaceMap )
        params.outVoxelPerFaceMap->clear();
    for ( auto& [ind, t, faceMap] : resTriangulatoinData )
    {
        resTriangulation.vec_.insert( resTriangulation.vec_.end(),
            std::make_move_iterator( t.vec_.begin() ), std::make_move_iterator( t.vec_.end() ) );
        if ( params.outVoxelPerFaceMap )
            params.outVoxelPerFaceMap->vec_.insert( params.outVoxelPerFaceMap->vec_.end(),
                std::make_move_iterator( faceMap.vec_.begin() ), std::make_move_iterator( faceMap.vec_.end() ) );
    }
    result.topology = MeshBuilder::fromTriangles( std::move( resTriangulation ) );
    result.points.resize( result.topology.lastValidVert() + 1 );
    assert( result.points.size() == totalVertices );

    if ( params.cb && !params.cb( 0.95f ) )
        return unexpectedOperationCanceled();

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, hmaps.size(), 1 ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        for ( auto& [_, set] : hmaps[range.begin()] )
        {
            for ( int i = int( NeighborDir::X ); i < int( NeighborDir::Count ); ++i )
                if ( set[i].vid.valid() )
                    result.points[set[i].vid] = set[i].position;
        }
    } );

    if ( params.cb && !params.cb( 1.0f ) )
        return unexpectedOperationCanceled();

    return result;
}

template <typename V, typename NaNChecker>
Expected<Mesh, std::string> volumeToMeshHelper1( const V& volume, const MarchingCubesParams& params, NaNChecker&& nanChecker )
{
    if ( !params.positioner )
        return volumeToMesh<V, NaNChecker, true>( volume, params, std::forward<NaNChecker>( nanChecker ) );
    else
        return volumeToMesh<V, NaNChecker, false>( volume, params, std::forward<NaNChecker>( nanChecker ) );
}

template <typename V>
Expected<Mesh, std::string> volumeToMeshHelper2( const V& volume, const MarchingCubesParams& params )
{
    if ( params.omitNaNCheck )
        return volumeToMeshHelper1( volume, params, [] ( float ) { return false; } );
    else
        return volumeToMeshHelper1( volume, params, isNanFast );
}

Expected<Mesh, std::string> marchingCubes( const SimpleVolume& volume, const MarchingCubesParams& params /*= {} */ )
{
    return volumeToMeshHelper2( volume, params );
}
Expected<Mesh, std::string> marchingCubes( const VdbVolume& volume, const MarchingCubesParams& params /*= {} */ )
{
    return volumeToMeshHelper2( volume, params );
}
Expected<Mesh, std::string> marchingCubes( const FunctionVolume& volume, const MarchingCubesParams& params )
{
    return volumeToMeshHelper2( volume, params );
}

Expected<Mesh, std::string> simpleVolumeToMesh( const SimpleVolume& volume, const MarchingCubesParams& params )
{
    return marchingCubes( volume, params );
}

Expected<Mesh, std::string> vdbVolumeToMesh( const VdbVolume& volume, const MarchingCubesParams& params )
{
    return marchingCubes( volume, params );
}

Expected<Mesh, std::string> functionVolumeToMesh( const FunctionVolume& volume, const MarchingCubesParams& params )
{
    return marchingCubes( volume, params );
}

}
#endif
