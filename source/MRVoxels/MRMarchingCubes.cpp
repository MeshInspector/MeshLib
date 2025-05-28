#include "MRMarchingCubes.h"
#include "MRVoxelsVolumeCachingAccessor.h"
#include "MROpenVDB.h"
#include "MRMesh/MRSeparationPoint.h"
#include "MRMesh/MRIsNaN.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRTriMesh.h"

#include <thread>

namespace MR
{

namespace
{

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

const std::array<OutEdge, size_t( NeighborDir::Count )> cPlusOutEdges { OutEdge::PlusX, OutEdge::PlusY, OutEdge::PlusZ };

class VolumeMesher
{
public:
    /// performs everything inside to convert volume into trimesh
    /// \param layersPerBlock all z-slices of the volume will be partitioned on blocks of given size to process in parallel (0 means auto-select layersPerBlock)
    template<typename V>
    static Expected<TriMesh> run( const V& volume, const MarchingCubesParams& params, int layersPerBlock = 0 );

public: // custom interface
    /// prepares convention for given volume dimensions and given parameters
    /// \param layersPerBlock all z-slices of the volume will be partitioned on blocks of given size to process in parallel (0 means auto-select layersPerBlock)
    explicit VolumeMesher( const Vector3i & dims, const MarchingCubesParams& params, int layersPerBlock );

    /// adds one more part of volume into consideration,
    template<typename V>
    Expected<void> addPart( const V& part );

    /// finishes processing and outputs produced trimesh
    Expected<TriMesh> finalize();

    int layersPerBlock() const { return layersPerBlock_; }
    int nextZ() const { return nextZ_; }

private:
    struct BlockInfo
    {
        int partFirstZ = 0;
        std::atomic<int>& numProcessedLayers;

        int blockIndex = 0;
        int layerBegin = 0;
        int layerEnd = 0;
        ProgressCallback myProgress;
        std::atomic<bool>* keepGoing = nullptr;
    };

    template<typename V>
    void addPartBlock_( const V& volume, const BlockInfo& blockInfo );
    void addBinaryPartBlock_( const SimpleBinaryVolume& volume, const BlockInfo& blockInfo );

private:
    VolumeIndexer indexer_;
    const MarchingCubesParams params_;
    int blockCount_ = 0;
    int layersPerBlock_ = 0;
    int nextZ_ = 0;

    std::vector<BitSet> invalids_; ///< invalid voxels in each layer
    std::vector<BitSet> lowerIso_; ///< voxels with the values lower then params.iso

    SeparationPointStorage sepStorage_;
};

template<typename V>
Expected<TriMesh> VolumeMesher::run( const V& volume, const MarchingCubesParams& params, int layersPerBlock )
{
    if ( volume.dims.x <= 0 || volume.dims.y <= 0 || volume.dims.z <= 0 )
        return TriMesh{};
    MR_TIMER;

    VolumeMesher mesher( volume.dims, params, layersPerBlock );
    return mesher.addPart( volume ).and_then( [&]
    {
        // free input volume, since it will not be used below any more
        if ( params.freeVolume )
            params.freeVolume();

        return mesher.finalize();
    } );
}

VolumeMesher::VolumeMesher( const Vector3i & dims, const MarchingCubesParams& params, int layersPerBlock ) : indexer_( dims ), params_( params )
{
    int threadCount = (int)tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism );
    if ( threadCount == 0 )
        threadCount = std::thread::hardware_concurrency();
    if ( threadCount == 0 )
        threadCount = 1;

    const int layerCount = indexer_.dims().z;
    const auto layerSize = indexer_.sizeXY();
    assert( indexer_.size() == layerCount * layerSize );

    // more blocks than threads is recommended for better work distribution among threads since
    // every block demands unique amount of processing
    if ( layersPerBlock <= 0 )
    {
        const auto approxBlockCount = std::min( layerCount, threadCount > 1 ? 4 * threadCount : 1 );
        layersPerBlock = (int)std::ceil( (float)layerCount / (float)approxBlockCount );
    }
    layersPerBlock_ = layersPerBlock;
    blockCount_ = ( layerCount + layersPerBlock_ - 1 ) / layersPerBlock_;
    const auto blockSize = layersPerBlock_ * layerSize;
    assert( indexer_.size() <= blockSize * blockCount_ );

    sepStorage_.resize( blockCount_, blockSize );

    invalids_.resize( layerCount );
    lowerIso_.resize( layerCount );
}

template<typename V>
Expected<void> VolumeMesher::addPart( const V& part )
{
    MR_TIMER;

    const int partFirstZ = nextZ_;
    if ( part.dims.x != indexer_.dims().x || part.dims.y != indexer_.dims().y )
        return unexpected( "XY dimensions of a part must be equal to XY dimensions of whole volume" );
    if ( part.dims.z <= 1 )
        return unexpected( "a part must have at least two Z slices" );
    if ( partFirstZ + part.dims.z > indexer_.dims().z )
        return unexpected( "a part exceeds whole volume in Z dimension" );
    const int layerCount = indexer_.dims().z;

    constexpr bool binary = std::is_same_v<V, SimpleBinaryVolume>;
    if constexpr ( binary )
    {
        int fillFirstZ = partFirstZ;
        if ( fillFirstZ )
            ++fillFirstZ; // skip already filled layer
        ParallelFor( fillFirstZ, fillFirstZ + part.dims.z, [&]( size_t z )
        {
            const auto layerSize = indexer_.sizeXY();
            BitSet layerLowerIso( layerSize );
            const auto firstLayerId = VoxelId( ( z - partFirstZ ) * layerSize );
            for ( size_t i = 0; i < layerSize; ++i )
                layerLowerIso.set( i, !part.data.test( firstLayerId + i ) );
            if ( layerLowerIso.any() )
                lowerIso_[z] = std::move( layerLowerIso );
        } );
    }

    const auto callingThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };

    // avoid false sharing with other local variables
    // by putting processedBits in its own cache line
    constexpr int hardware_destructive_interference_size = 64;
    struct alignas(hardware_destructive_interference_size) S
    {
        std::atomic<int> numProcessedLayers{ 0 };
    } cacheLineStorage;
    static_assert( alignof(S) == hardware_destructive_interference_size );
    static_assert( sizeof(S) == hardware_destructive_interference_size );

    const int firstBlock = partFirstZ / layersPerBlock_;
    nextZ_ = partFirstZ + part.dims.z - 1;
    const bool lastPart = nextZ_ + 1 == indexer_.dims().z;
    const int lastLayer = lastPart ? nextZ_ : nextZ_ - 1;
    assert( lastLayer < layerCount );
    const int lastBlock = lastLayer / layersPerBlock_;

    const auto cb = subprogress( params_.cb, 0.0f, 0.3f );
    auto currentSubprogress = subprogress(
        cb,
        (float)partFirstZ / (float)indexer_.dims().z,
        (float)lastLayer / (float)indexer_.dims().z
    );

    ParallelFor( firstBlock, lastBlock + 1, [&] ( int blockIndex )
    {
        BlockInfo blockInfo
        {
            .partFirstZ = partFirstZ,
            .numProcessedLayers = cacheLineStorage.numProcessedLayers,
            .blockIndex = blockIndex
        };
        blockInfo.layerBegin = std::max( blockIndex * layersPerBlock_, partFirstZ );
        if ( blockInfo.layerBegin >= layerCount )
            return;
        blockInfo.layerEnd = std::min( ( blockIndex + 1 ) * layersPerBlock_, lastLayer + 1 );

        if ( currentSubprogress )
        {
            blockInfo.keepGoing = &keepGoing;
            if ( std::this_thread::get_id() == callingThreadId )
            {
                // from dedicated thread only: actually report progress proportional to the number of processed layers
                blockInfo.myProgress = [&]( float ) // input value is ignored
                {
                    auto l = cacheLineStorage.numProcessedLayers.load( std::memory_order_relaxed );
                    bool res = currentSubprogress( float( l ) / layerCount );
                    if ( !res )
                        keepGoing.store( false, std::memory_order_relaxed );
                    return res;
                };
            }
            else // from other threads just check that the operation was not canceled
                blockInfo.myProgress = [&]( float ) { return keepGoing.load( std::memory_order_relaxed ); };
        }

        if constexpr ( binary )
            addBinaryPartBlock_( part, blockInfo );
        else
            addPartBlock_( part, blockInfo );
    } );

    if ( currentSubprogress && !keepGoing )
        return unexpectedOperationCanceled();

    return {};
}

template<typename V>
void VolumeMesher::addPartBlock_( const V& part, const BlockInfo& blockInfo )
{
    MR_TIMER;
    auto cachingMode = params_.cachingMode;
    if ( cachingMode == MarchingCubesParams::CachingMode::Automatic )
    {
        if constexpr ( VoxelsVolumeAccessor<V>::cacheEffective )
            cachingMode = MarchingCubesParams::CachingMode::Normal;
        else
            cachingMode = MarchingCubesParams::CachingMode::None;
    }

    auto & block = sepStorage_.getBlock( blockInfo.blockIndex );
    const auto layerSize = indexer_.sizeXY();
    const auto partFirstId = layerSize * blockInfo.partFirstZ;
    const VolumeIndexer partIndexer( part.dims );
    const VoxelsVolumeAccessor<V> acc( part );
    /// grid point of this part with integer coordinates (0,0,0) will be shifted to this position in 3D space
    const Vector3f zeroPoint = params_.origin + mult( acc.shift() + Vector3f( 0, 0, (float)blockInfo.partFirstZ ), part.voxelSize );

    auto positioner = [this]( const Vector3f& pos0, const Vector3f& pos1, float v0, float v1, float iso )
    {
        if ( params_.positioner )
            return params_.positioner( pos0, pos1, v0, v1, iso );
        assert( v0 != v1 );
        const auto ratio = ( iso - v0 ) / ( v1 - v0 );
        assert( ratio >= 0 && ratio <= 1 );
        return ( 1.0f - ratio ) * pos0 + ratio * pos1;
    };

    std::optional<VoxelsVolumeCachingAccessor<V>> cache;
    if ( cachingMode == MarchingCubesParams::CachingMode::Normal )
    {
        using Parameters = typename VoxelsVolumeCachingAccessor<V>::Parameters;
        cache.emplace( acc, partIndexer, Parameters {
            .preloadedLayerCount = 2,
        } );
        if ( !cache->preloadLayer( blockInfo.layerBegin - blockInfo.partFirstZ, blockInfo.myProgress ) )
            return;
    }

    VoxelLocation loc = partIndexer.toLoc( Vector3i( 0, 0, blockInfo.layerBegin - blockInfo.partFirstZ ) );
    for ( ; loc.pos.z + blockInfo.partFirstZ < blockInfo.layerEnd; ++loc.pos.z )
    {
        if ( cache && loc.pos.z != cache->currentLayer() )
        {
            if ( !cache->preloadNextLayer( blockInfo.myProgress ) )
                return;
            assert( loc.pos.z == cache->currentLayer() );
        }
        BitSet layerInvalids( layerSize );
        BitSet layerLowerIso( layerSize );
        size_t inLayerPos = 0;
        for ( loc.pos.y = 0; loc.pos.y < part.dims.y; ++loc.pos.y )
        {
            for ( loc.pos.x = 0; loc.pos.x < part.dims.x; ++loc.pos.x, ++loc.id, ++inLayerPos )
            {
                assert( partIndexer.toVoxelId( loc.pos ) == loc.id );
                if ( blockInfo.keepGoing && !blockInfo.keepGoing->load( std::memory_order_relaxed ) )
                    return;

                SeparationPointSet set;
                bool atLeastOneOk = false;
                const float value = cache ? cache->get( loc ) : acc.get( loc );
                const bool lower = value < params_.iso;
                const bool notLower = value >= params_.iso;
                if ( !lower && !notLower ) // both not-lower and not-same-or-higher can be true only if value is not-a-number (NaN)
                    layerInvalids.set( inLayerPos );
                else
                {
                    const auto coords = zeroPoint + mult( part.voxelSize, Vector3f( loc.pos ) );
                    layerLowerIso.set( inLayerPos, lower );

                    for ( int n = int( NeighborDir::X ); n < int( NeighborDir::Count ); ++n )
                    {
                        auto nextLoc = partIndexer.getNeighbor( loc, cPlusOutEdges[n] );
                        if ( !nextLoc )
                            continue;
                        const float nextValue = cache ? cache->get( nextLoc ) : acc.get( nextLoc );
                        if ( lower )
                        {
                            if ( !( nextValue >= params_.iso ) )
                                continue; // nextValue is lower than params_.iso (same as value) or nextValue is NaN
                        }
                        else
                        {
                            if ( !( nextValue < params_.iso ) )
                                continue; // nextValue is same or higher than params_.iso (same as value) or nextValue is NaN
                        }

                        auto nextCoords = coords;
                        nextCoords[n] += part.voxelSize[n];
                        Vector3f pos = positioner( coords, nextCoords, value, nextValue, params_.iso );
                        set[n] = block.nextVid();
                        block.coords.push_back( pos );
                        atLeastOneOk = true;
                    }
                }

                if ( !atLeastOneOk )
                    continue;

                block.smap.insert( { loc.id + partFirstId, set } );
            }
        }
        if ( layerInvalids.any() )
            invalids_[loc.pos.z + blockInfo.partFirstZ] = std::move( layerInvalids );
        if ( layerLowerIso.any() )
            lowerIso_[loc.pos.z + blockInfo.partFirstZ] = std::move( layerLowerIso );
        blockInfo.numProcessedLayers.fetch_add( 1, std::memory_order_relaxed );
        if ( !reportProgress( blockInfo.myProgress, 1.f ) ) // 1. is ignored anyway
            return;
    }
}

void VolumeMesher::addBinaryPartBlock_( const SimpleBinaryVolume& part, const BlockInfo& blockInfo )
{
    MR_TIMER;

    auto & block = sepStorage_.getBlock( blockInfo.blockIndex );
    const auto layerSize = indexer_.sizeXY();
    const auto partFirstId = layerSize * blockInfo.partFirstZ;
    const VolumeIndexer partIndexer( part.dims );
    /// grid point of this part with integer coordinates (0,0,0) will be shifted to this position in 3D space
    const Vector3f zeroPoint = params_.origin + mult( Vector3f::diagonal( 0.5f ) + Vector3f( 0, 0, (float)blockInfo.partFirstZ ), part.voxelSize );

    auto positioner = [this]( const Vector3f& pos0, const Vector3f& pos1, float iso )
    {
        if ( params_.positioner )
            return params_.positioner( pos0, pos1, 0.f, 1.f, iso );
        return ( 1.0f - iso ) * pos0 + iso * pos1;
    };

    VoxelLocation loc = partIndexer.toLoc( Vector3i( 0, 0, blockInfo.layerBegin - blockInfo.partFirstZ ) );
    for ( ; loc.pos.z + blockInfo.partFirstZ < blockInfo.layerEnd; ++loc.pos.z )
    {
        const auto layerZ = loc.pos.z + blockInfo.partFirstZ;
        const auto& layerLowerIso = lowerIso_[layerZ];
        const auto* nextLayerLowerIso = layerZ + 1 < lowerIso_.size() ? &lowerIso_[layerZ + 1] : nullptr;
        size_t inLayerPos = 0;
        for ( loc.pos.y = 0; loc.pos.y < part.dims.y; ++loc.pos.y )
        {
            for ( loc.pos.x = 0; loc.pos.x < part.dims.x; ++loc.pos.x, ++loc.id, ++inLayerPos )
            {
                assert( partIndexer.toVoxelId( loc.pos ) == loc.id );
                if ( blockInfo.keepGoing && !blockInfo.keepGoing->load( std::memory_order_relaxed ) )
                    return;

                SeparationPointSet set;
                const auto size0 = block.coords.size();
                const bool lower = layerLowerIso.test( inLayerPos );
                const auto coords = zeroPoint + mult( part.voxelSize, Vector3f( loc.pos ) );

                auto addPoint = [&]( int n )
                {
                    auto nextCoords = coords;
                    nextCoords[n] += part.voxelSize[n];
                    Vector3f pos = lower ? positioner( coords, nextCoords, params_.iso )
                                         : positioner( nextCoords, coords, params_.iso );
                    set[n] = block.nextVid();
                    block.coords.push_back( pos );
                };

                if ( loc.pos.x + 1 < part.dims.x && lower != layerLowerIso.test( inLayerPos + 1 ) )
                    addPoint( 0 );
                if ( loc.pos.y + 1 < part.dims.y && lower != layerLowerIso.test( inLayerPos + part.dims.x ) )
                    addPoint( 1 );
                if ( nextLayerLowerIso && lower != nextLayerLowerIso->test( inLayerPos ) )
                    addPoint( 2 );
                if ( size0 == block.coords.size() )
                    continue;

                block.smap.insert( { loc.id + partFirstId, set } );
            }
        }
        blockInfo.numProcessedLayers.fetch_add( 1, std::memory_order_relaxed );
        if ( !reportProgress( blockInfo.myProgress, 1.f ) ) // 1. is ignored anyway
            return;
    }
}

Expected<TriMesh> VolumeMesher::finalize()
{
    MR_TIMER;
    if ( nextZ_ + 1 != indexer_.dims().z )
        return unexpected( "Provided parts do not cover whole volume" );

    const auto totalVertices = sepStorage_.makeUniqueVids();
    if ( totalVertices > params_.maxVertices )
        return unexpected( "Vertices number limit exceeded." );

    if ( params_.cb && !params_.cb( 0.5f ) )
        return unexpectedOperationCanceled();

    const size_t dimsX = indexer_.dims().x;
    const size_t cVoxelNeighborsIndexAdd[8] =
    {
        0,
        1,
        dimsX,
        dimsX + 1,
        indexer_.sizeXY(),
        indexer_.sizeXY() + 1,
        indexer_.sizeXY() + dimsX,
        indexer_.sizeXY() + dimsX + 1
    };
    const size_t cDimStep[3] = { 1, dimsX, indexer_.sizeXY() };

    const bool hasInvalidVoxels =
        std::any_of( invalids_.begin(), invalids_.end(), []( const BitSet & bs ) { return !bs.empty(); } ); // bit set is not empty only if at least one bit is set

    const auto callingThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };

    // avoid false sharing with other local variables
    // by putting processedBits in its own cache line
    constexpr int hardware_destructive_interference_size = 64;
    struct alignas(hardware_destructive_interference_size) S
    {
        std::atomic<int> numProcessedLayers{ 0 };
    } cacheLineStorage;
    static_assert( alignof(S) == hardware_destructive_interference_size );
    static_assert( sizeof(S) == hardware_destructive_interference_size );

    const int layerCount = indexer_.dims().z;
    auto currentSubprogress = subprogress( params_.cb, 0.5f, 0.85f );
    ParallelFor( 0, blockCount_, [&] ( int blockIndex )
    {
        auto & block = sepStorage_.getBlock( blockIndex );
        const bool report = currentSubprogress && std::this_thread::get_id() == callingThreadId;

        const int layerBegin = blockIndex * layersPerBlock_;
        if ( layerBegin >= layerCount )
            return;
        const auto layerEnd = std::min( ( blockIndex + 1 ) * layersPerBlock_, layerCount - 1 ); // skip last layer since no data from next layer

        // cell data
        std::array<const SeparationPointSet*, 7> neis;
        unsigned char voxelConfiguration;
        VoxelLocation loc = indexer_.toLoc( Vector3i( 0, 0, layerBegin ) );
        for ( ; loc.pos.z < layerEnd; ++loc.pos.z )
        {
            const BitSet* layerInvalids[2] = { &invalids_[loc.pos.z], &invalids_[loc.pos.z+1] };
            const BitSet* layerLowerIso[2] = { &lowerIso_[loc.pos.z], &lowerIso_[loc.pos.z+1] };
            const VoxelId layerFirstVoxelId[2] = { indexer_.toVoxelId( { 0, 0, loc.pos.z } ), indexer_.toVoxelId( { 0, 0, loc.pos.z + 1 } ) };
            // returns a bit from from one-of-two bit sets (bs) corresponding to given location (vl)
            auto getBit = [&]( const BitSet *bs[2], const VoxelLocation & vl )
            {
                const auto dl = vl.pos.z - loc.pos.z;
                assert( dl >= 0 && dl <= 1 );
                // (*bs)[dl] is one of two bit sets, and layerFirstVoxelId[dl] is VoxelId corresponding to zeroth bit in it
                return (*bs)[dl].test( vl.id - layerFirstVoxelId[dl] );
            };
            for ( loc.pos.y = 0; loc.pos.y + 1 < indexer_.dims().y; ++loc.pos.y )
            {
                loc.pos.x = 0;
                loc.id = indexer_.toVoxelId( loc.pos );
                auto posXY = dimsX * loc.pos.y;
                for ( ; loc.pos.x + 1 < dimsX; ++loc.pos.x, ++loc.id, ++posXY )
                {
                    assert( indexer_.toVoxelId( loc.pos ) == loc.id );
                    if ( params_.cb && !keepGoing.load( std::memory_order_relaxed ) )
                        return;

                    bool voxelValid = true;
                    voxelConfiguration = 0;
                    bool vx[8] =
                    {
                        layerLowerIso[0]->test( posXY ),
                        layerLowerIso[0]->test( posXY + 1 ),
                        layerLowerIso[0]->test( posXY + dimsX ),
                        layerLowerIso[0]->test( posXY + dimsX + 1 ),
                        layerLowerIso[1]->test( posXY ),
                        layerLowerIso[1]->test( posXY + 1 ),
                        layerLowerIso[1]->test( posXY + dimsX ),
                        layerLowerIso[1]->test( posXY + dimsX + 1 )
                    };
                    [[maybe_unused]] bool atLeastOneNan = false;
                    for ( int i = 0; i < cVoxelNeighbors.size(); ++i )
                    {
                        bool voxelValueLowerIso = vx[i]; //faster alternative of getBit( layerLowerIso, nloc );
                        if ( hasInvalidVoxels )
                        {
                            VoxelLocation nloc{ loc.id + cVoxelNeighborsIndexAdd[i], loc.pos + cVoxelNeighbors[i] };
                            bool invalidVoxelValue = getBit( layerInvalids, nloc );
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
                            while ( invalidVoxelValue && neighIndex < 7 )
                            {
                                auto neighLoc = nloc;
                                for ( int posCoord = 0; posCoord < 3; ++posCoord )
                                {
                                    if ( !( ( cNeighborsOrder[neighIndex] & ( 1 << posCoord ) ) >> posCoord ) )
                                        continue;
                                    if ( cVoxelNeighbors[i][posCoord] == 1 )
                                    {
                                        --neighLoc.pos[posCoord];
                                        neighLoc.id -= cDimStep[posCoord];
                                    }
                                    else
                                    {
                                        ++neighLoc.pos[posCoord];
                                        neighLoc.id += cDimStep[posCoord];
                                    }
                                }
                                invalidVoxelValue = getBit( layerInvalids, neighLoc );
                                voxelValueLowerIso = getBit( layerLowerIso, neighLoc );
                                ++neighIndex;
                            }
                            if ( invalidVoxelValue )
                            {
                                voxelValid = false;
                                break;
                            }
                            if ( !atLeastOneNan && neighIndex > 0 )
                                atLeastOneNan = true;
                            vx[i] = voxelValueLowerIso;
                        }
                        if ( voxelValueLowerIso )
                            voxelConfiguration |= cMapNeighbors[i];
                    }
                    if ( !voxelValid || voxelConfiguration == 0x00 || voxelConfiguration == 0xff )
                        continue;

                    // find only necessary neighbor separation points by comparing
                    // voxel values in both ends of each edge relative params_.iso (stored in vx array);
                    // separation points will not be used (and can be not searched for better performance)
                    // if both ends of the edge are higher or both are lower than params_.iso
                    voxelValid = false;
                    auto findNei = [&]( int i, auto check )
                    {
                        const auto index = loc.id + cVoxelNeighborsIndexAdd[i];
                        auto * pSet = sepStorage_.findSeparationPointSet( index );
                        if ( pSet && check( *pSet ) )
                        {
                            neis[i] = pSet;
                            voxelValid = true;
                        }
                    };

                    neis = {};
                    if ( vx[0] != vx[1] || vx[0] != vx[2] || vx[0] != vx[4] )
                        findNei( 0, []( auto && ) { return true; } );
                    if ( vx[1] != vx[3] || vx[1] != vx[5] )
                        findNei( 1, []( auto && s ) { return s[(int)NeighborDir::Y] || s[(int)NeighborDir::Z]; } );
                    if ( vx[2] != vx[3] || vx[2] != vx[6] )
                        findNei( 2, []( auto && s ) { return s[(int)NeighborDir::X] || s[(int)NeighborDir::Z]; } );
                    if ( vx[3] != vx[7] )
                        findNei( 3, []( auto && s ) { return (bool)s[(int)NeighborDir::Z]; } );
                    if ( vx[4] != vx[5] || vx[4] != vx[6] )
                        findNei( 4, []( auto && s ) { return s[(int)NeighborDir::X] || s[(int)NeighborDir::Y]; } );
                    if ( vx[5] != vx[7] )
                        findNei( 5, []( auto && s ) { return (bool)s[(int)NeighborDir::Y]; } );
                    if ( vx[6] != vx[7] )
                        findNei( 6, []( auto && s ) { return (bool)s[(int)NeighborDir::X]; } );

                    // ensure consistent nan voxel
                    if ( atLeastOneNan && voxelValid )
                    {
                        const auto& plan = cTriangleTable[voxelConfiguration];
                        for ( int i = 0; i < plan.size() && voxelValid; i += 3 )
                        {
                            const auto& [interIndex0, dir0] = cEdgeIndicesMap[plan[i]];
                            const auto& [interIndex1, dir1] = cEdgeIndicesMap[plan[i + 1]];
                            const auto& [interIndex2, dir2] = cEdgeIndicesMap[plan[i + 2]];
                            // `neis` indicates that current voxel has valid point for desired triangulation
                            // as far as nei has 3 directions we use `dir` to validate (make sure that there is point in needed edge) desired direction
                            voxelValid = voxelValid && neis[interIndex0] && (*neis[interIndex0])[int( dir0 )];
                            voxelValid = voxelValid && neis[interIndex1] && (*neis[interIndex1])[int( dir1 )];
                            voxelValid = voxelValid && neis[interIndex2] && (*neis[interIndex2])[int( dir2 )];
                        }
                    }
                    if ( !voxelValid )
                        continue;

                    const auto& plan = cTriangleTable[voxelConfiguration];
                    for ( int i = 0; i < plan.size(); i += 3 )
                    {
                        const auto& [interIndex0, dir0] = cEdgeIndicesMap[plan[i]];
                        const auto& [interIndex1, dir1] = cEdgeIndicesMap[plan[i + 1]];
                        const auto& [interIndex2, dir2] = cEdgeIndicesMap[plan[i + 2]];
                        assert( neis[interIndex0] && (*neis[interIndex0])[int( dir0 )] );
                        assert( neis[interIndex1] && (*neis[interIndex1])[int( dir1 )] );
                        assert( neis[interIndex2] && (*neis[interIndex2])[int( dir2 )] );

                        if ( params_.lessInside )
                            block.tris.emplace_back( ThreeVertIds{
                                (*neis[interIndex0])[int( dir0 )],
                                (*neis[interIndex2])[int( dir2 )],
                                (*neis[interIndex1])[int( dir1 )]
                            } );
                        else
                            block.tris.emplace_back( ThreeVertIds{
                                (*neis[interIndex0])[int( dir0 )],
                                (*neis[interIndex1])[int( dir1 )],
                                (*neis[interIndex2])[int( dir2 )]
                            } );
                        if ( params_.outVoxelPerFaceMap )
                            block.faceMap.emplace_back( loc.id );
                    }
                }
            }
            // free memory containing unused data
            if ( loc.pos.z > layerBegin || loc.pos.z == 0 ) // processed layer, not the first in the block (or the first in the first block)
            {
                invalids_[loc.pos.z] = {};
                lowerIso_[loc.pos.z] = {};
            }
            if ( loc.pos.z + 2 == layerCount ) // the very last layer after this one
            {
                invalids_[loc.pos.z + 1] = {};
                lowerIso_[loc.pos.z + 1] = {};
            }

            const auto numProcessedLayers = 1 + cacheLineStorage.numProcessedLayers.fetch_add( 1, std::memory_order_relaxed );
            if ( report && !reportProgress( currentSubprogress, float( numProcessedLayers ) / layerCount ) )
            {
                keepGoing.store( false, std::memory_order_relaxed );
                return;
            }
        }
    } );

    if ( params_.cb && !keepGoing )
        return unexpectedOperationCanceled();

    // no longer needed, reduce peak memory consumption
    invalids_ = {};
    lowerIso_ = {};

    // create result triangulation
    TriMesh result;
    result.tris = sepStorage_.getTriangulation( params_.outVoxelPerFaceMap );

    if ( params_.cb && !params_.cb( 0.95f ) )
        return unexpectedOperationCanceled();

    // some points may be not referenced by any triangle due to NaNs
    result.points.resize( totalVertices );
    sepStorage_.getPoints( result.points );

    if ( params_.cb && !params_.cb( 1.0f ) )
        return unexpectedOperationCanceled();

    return result;
}

} // anonymous namespace

Expected<TriMesh> marchingCubesAsTriMesh( const SimpleVolume& volume, const MarchingCubesParams& params /*= {} */ )
{
    return VolumeMesher::run( volume, params );
}

Expected<Mesh> marchingCubes( const SimpleVolume& volume, const MarchingCubesParams& params )
{
    MR_TIMER;
    auto p = params;
    p.cb = subprogress( params.cb, 0.0f, 0.9f );
    return marchingCubesAsTriMesh( volume, p ).and_then( [&params]( TriMesh && tm ) -> Expected<Mesh>
    {
        return Mesh::fromTriMesh( std::move( tm ), {}, subprogress( params.cb, 0.9f, 1.0f ) );
    } );
}

Expected<TriMesh> marchingCubesAsTriMesh( const SimpleVolumeMinMax& volume, const MarchingCubesParams& params /*= {} */ )
{
    if ( params.iso <= volume.min || params.iso >= volume.max )
        return TriMesh{};
    return VolumeMesher::run( volume, params );
}

Expected<Mesh> marchingCubes( const SimpleVolumeMinMax& volume, const MarchingCubesParams& params )
{
    MR_TIMER;
    auto p = params;
    p.cb = subprogress( params.cb, 0.0f, 0.9f );
    return marchingCubesAsTriMesh( volume, p ).and_then( [&params]( TriMesh && tm ) -> Expected<Mesh>
    {
        return Mesh::fromTriMesh( std::move( tm ), {}, subprogress( params.cb, 0.9f, 1.0f ) );
    } );
}

Expected<TriMesh> marchingCubesAsTriMesh( const VdbVolume& volume, const MarchingCubesParams& params /*= {} */ )
{
    if ( !volume.data )
        return unexpected( "No volume data." );
    if ( params.iso <= volume.min || params.iso >= volume.max )
        return TriMesh{};
    return VolumeMesher::run( volume, params );
}

Expected<Mesh> marchingCubes( const VdbVolume& volume, const MarchingCubesParams& params /*= {} */ )
{
    MR_TIMER;
    auto p = params;
    p.cb = subprogress( params.cb, 0.0f, 0.9f );
    return marchingCubesAsTriMesh( volume, p ).and_then( [&params]( TriMesh && tm ) -> Expected<Mesh>
    {
        return Mesh::fromTriMesh( std::move( tm ), {}, subprogress( params.cb, 0.9f, 1.0f ) );
    } );
}

Expected<TriMesh> marchingCubesAsTriMesh( const FunctionVolume& volume, const MarchingCubesParams& params )
{
    if ( !volume.data )
        return unexpected( "Getter function is not specified." );
    return VolumeMesher::run( volume, params );
}

Expected<Mesh> marchingCubes( const FunctionVolume& volume, const MarchingCubesParams& params )
{
    MR_TIMER;
    auto p = params;
    p.cb = subprogress( params.cb, 0.0f, 0.9f );
    return marchingCubesAsTriMesh( volume, p ).and_then( [&params]( TriMesh && tm ) -> Expected<Mesh>
    {
        return Mesh::fromTriMesh( std::move( tm ), {}, subprogress( params.cb, 0.9f, 1.0f ) );
    } );
}

Expected<TriMesh> marchingCubesAsTriMesh( const SimpleBinaryVolume& volume, const MarchingCubesParams& params /*= {} */ )
{
    if ( params.iso <= 0 || params.iso >= 1 )
        return TriMesh{};
    return VolumeMesher::run( volume, params );
}

Expected<Mesh> marchingCubes( const SimpleBinaryVolume& volume, const MarchingCubesParams& params )
{
    MR_TIMER;
    auto p = params;
    p.cb = subprogress( params.cb, 0.0f, 0.9f );
    return marchingCubesAsTriMesh( volume, p ).and_then( [&params]( TriMesh && tm ) -> Expected<Mesh>
    {
        return Mesh::fromTriMesh( std::move( tm ), {}, subprogress( params.cb, 0.9f, 1.0f ) );
    } );
}

struct MarchingCubesByParts::Impl
{
    VolumeMesher mesher;
};

MarchingCubesByParts::MarchingCubesByParts( const Vector3i & dims, const MarchingCubesParams& params, int layersPerBlock )
    : impl_( new Impl{ VolumeMesher( dims, params, layersPerBlock ) } )
{
}

MarchingCubesByParts::~MarchingCubesByParts() = default;
MarchingCubesByParts::MarchingCubesByParts( MarchingCubesByParts && s ) noexcept = default;
MarchingCubesByParts & MarchingCubesByParts::operator=( MarchingCubesByParts && s ) noexcept = default;

int MarchingCubesByParts::layersPerBlock() const
{
    return impl_->mesher.layersPerBlock();
}

int MarchingCubesByParts::nextZ() const
{
    return impl_->mesher.nextZ();
}

Expected<void> MarchingCubesByParts::addPart( const SimpleVolume& part )
{
    return impl_->mesher.addPart( part );
}

Expected<TriMesh> MarchingCubesByParts::finalize()
{
    return impl_->mesher.finalize();
}

} //namespace MR
