#include "MRMarchingCubes.h"
#include "MRSeparationPoint.h"
#include "MRIsNaN.h"
#include "MRMesh.h"
#include "MRVolumeIndexer.h"
#include "MRVoxelsVolumeAccess.h"
#include "MRLine3.h"
#include "MRMeshBuilder.h"
#include "MRVDBFloatGrid.h"
#include "MRTimer.h"
#include "MRParallelFor.h"
#include "MRTriMesh.h"
#ifndef MRMESH_NO_OPENVDB
#include "MRPch/MROpenvdb.h"
#endif
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

}

template<typename V, typename NaNChecker, typename Positioner>
Expected<TriMesh> volumeToMesh( const V& volume, const MarchingCubesParams& params, NaNChecker&& nanChecker, Positioner&& positioner )
{
    TriMesh result;
    if ( volume.dims.x <= 0 || volume.dims.y <= 0 || volume.dims.z <= 0 )
        return result;

    MR_TIMER

    auto cachingMode = params.cachingMode;
    if ( cachingMode == MarchingCubesParams::CachingMode::Automatic )
    {
        if constexpr ( std::is_same_v<V, FunctionVolume> || std::is_same_v<V, FunctionVolumeU8> )
            cachingMode = MarchingCubesParams::CachingMode::Normal;
        else
            cachingMode = MarchingCubesParams::CachingMode::None;
    }

    VolumeIndexer indexer( volume.dims );

    std::atomic<bool> keepGoing{ true };
    auto mainThreadId = std::this_thread::get_id();
    int lastSubMap = -1;

    size_t threadCount = tbb::global_control::parameter( tbb::global_control::max_allowed_parallelism );
    if ( threadCount == 0 )
        threadCount = std::thread::hardware_concurrency();
    if ( threadCount == 0 )
        threadCount = 1;

    const auto layerCount = (size_t)indexer.dims().z;
    const auto layerSize = indexer.sizeXY();
    assert( indexer.size() == layerCount * layerSize );

    // more blocks than threads is recommended for better work distribution among threads since
    // every block demands unique amount of processing
    const auto blockCount = std::min( layerCount, threadCount > 1 ? 4 * threadCount : 1 );
    const auto layerPerBlockCount = (size_t)std::ceil( (float)layerCount / (float)blockCount );
    const auto blockSize = layerPerBlockCount * layerSize;
    assert( indexer.size() <= blockSize * blockCount );

    SeparationPointStorage sepStorage( blockCount, blockSize );

    ParallelFor( size_t( 0 ), blockCount, [&] ( size_t blockIndex )
    {
        auto & block = sepStorage.getBlock( blockIndex );

        if ( std::this_thread::get_id() == mainThreadId && lastSubMap == -1 )
            lastSubMap = int( blockIndex );
        const bool runCallback = params.cb && std::this_thread::get_id() == mainThreadId && lastSubMap == blockIndex;

        const auto layerBegin = blockIndex * layerPerBlockCount;
        if ( layerBegin >= layerCount )
            return;
        const auto layerEnd = std::min( ( blockIndex + 1 ) * layerPerBlockCount, layerCount );

        const VoxelsVolumeAccessor<V> acc( volume );
        /// grid point with integer coordinates (0,0,0) will be shifted to this position in 3D space
        const Vector3f zeroPoint = params.origin + mult( acc.shift(), volume.voxelSize );

        std::optional<VoxelsVolumeCachingAccessor<V>> cache;
        if ( cachingMode == MarchingCubesParams::CachingMode::Normal )
        {
            using Parameters = typename VoxelsVolumeCachingAccessor<V>::Parameters;
            cache.emplace( acc, indexer, Parameters {
                .preloadedLayerCount = 2,
            } );
            cache->preloadLayer( (int)layerBegin );
        }

        const auto begin = layerBegin * layerSize;
        const auto end = layerEnd * layerSize;

        for ( size_t i = begin; i < end; ++i )
        {
            if ( params.cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            const auto baseLoc = indexer.toLoc( VoxelId( i ) );
            if ( cache && baseLoc.pos.z != cache->currentLayer() )
            {
                cache->preloadNextLayer();
                assert( baseLoc.pos.z == cache->currentLayer() );
            }

            SeparationPointSet set;
            bool atLeastOneOk = false;
            const float baseValue = cache ? cache->get( baseLoc ) : acc.get( baseLoc );
            if ( !nanChecker( baseValue ) )
            {
                const auto baseCoords = zeroPoint + mult( volume.voxelSize, Vector3f( baseLoc.pos ) );
                const bool baseLower = baseValue < params.iso;

                for ( int n = int( NeighborDir::X ); n < int( NeighborDir::Count ); ++n )
                {
                    auto nextLoc = baseLoc;
                    nextLoc.pos[n] += 1;
                    if ( nextLoc.pos[n] >= indexer.dims()[n] )
                        continue;
                    nextLoc.id = indexer.getExistingNeighbor( baseLoc.id, cPlusOutEdges[n] );
                    const float nextValue = cache ? cache->get( nextLoc ) : acc.get( nextLoc );
                    if ( nanChecker( nextValue ) )
                        continue;

                    const bool nextLower = nextValue < params.iso;
                    if ( baseLower == nextLower )
                        continue;

                    auto nextCoords = baseCoords;
                    nextCoords[n] += volume.voxelSize[n];
                    Vector3f pos = positioner( baseCoords, nextCoords, baseValue, nextValue, params.iso );
                    set[n] = block.nextVid();
                    block.coords.push_back( pos );
                    atLeastOneOk = true;
                }
            }

            if ( runCallback && ( i - begin ) % 16384 == 0 )
                if ( !params.cb( 0.3f * float( i - begin ) / float( end - begin ) ) )
                    keepGoing.store( false, std::memory_order_relaxed );

            if ( !atLeastOneOk )
                continue;

            block.smap.insert( { i, set } );
        }
    } );

    if ( params.cb && !keepGoing )
        return unexpectedOperationCanceled();

    const auto totalVertices = sepStorage.makeUniqueVids();
    if ( totalVertices > params.maxVertices )
        return unexpected( "Vertices number limit exceeded." );

    if ( params.cb && !params.cb( 0.5f ) )
        return unexpectedOperationCanceled();

    auto subprogress2 = MR::subprogress( params.cb, 0.5f, 0.85f );

    const size_t cVoxelNeighborsIndexAdd[8] = 
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
    const size_t cDimStep[3] = { 1, size_t( indexer.dims().x ), indexer.sizeXY() };

    ParallelFor( size_t( 0 ), blockCount, [&] ( size_t blockIndex )
    {
        auto & block = sepStorage.getBlock( blockIndex );
        const auto layerBegin = blockIndex * layerPerBlockCount;
        if ( layerBegin >= layerCount )
            return;
        const auto layerEnd = std::min( ( blockIndex + 1 ) * layerPerBlockCount, layerCount );

        const VoxelsVolumeAccessor<V> acc( volume );
        std::optional<VoxelsVolumeCachingAccessor<V>> cache;
        if ( cachingMode == MarchingCubesParams::CachingMode::Normal )
        {
            using Parameters = typename VoxelsVolumeCachingAccessor<V>::Parameters;
            cache.emplace( acc, indexer, Parameters {
                .preloadedLayerCount = 2,
            } );
            cache->preloadLayer( (int)layerBegin );
        }

        const auto begin = layerBegin * layerSize;
        const auto end = layerEnd * layerSize;

        const bool runCallback = subprogress2 && std::this_thread::get_id() == mainThreadId;

        // cell data
        std::array<const SeparationPointSet*, 7> neis;
        unsigned char voxelConfiguration;
        for ( size_t ind = begin; ind < end; ++ind )
        {
            if ( subprogress2 && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            const auto baseLoc = indexer.toLoc( VoxelId( ind ) );
            if ( baseLoc.pos.x + 1 >= volume.dims.x ||
                baseLoc.pos.y + 1 >= volume.dims.y ||
                baseLoc.pos.z + 1 >= volume.dims.z )
                continue;

            if ( cache && baseLoc.pos.z != cache->currentLayer() )
            {
                cache->preloadNextLayer();
                assert( baseLoc.pos.z == cache->currentLayer() );
            }

            bool voxelValid = true;
            voxelConfiguration = 0;
            std::array<bool, 8> vx{};
            [[maybe_unused]] bool atLeastOneNan = false;
            for ( int i = 0; i < cVoxelNeighbors.size(); ++i )
            {
                VoxelLocation loc{ baseLoc.id + cVoxelNeighborsIndexAdd[i], baseLoc.pos + cVoxelNeighbors[i] };
                float value = cache ? cache->get( loc ) : acc.get( loc );
                if ( !params.omitNaNCheck )
                {
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
                        auto neighLoc = loc;
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
                        value = cache ? cache->get( neighLoc ) : acc.get( neighLoc );
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
                
                if ( value >= params.iso )
                    continue;
                voxelConfiguration |= cMapNeighbors[i];
                vx[i] = true;
            }
            if ( !voxelValid || voxelConfiguration == 0x00 || voxelConfiguration == 0xff )
                continue;

            // find only necessary neighbor separation points by comparing
            // voxel values in both ends of each edge relative iso (stored in vx array);
            // separation points will not be used (and can be not searched for better performance)
            // if both ends of the edge are higher or both are lower than iso
            voxelValid = false;
            auto findNei = [&]( int i, auto check )
            {
                const auto index = ind + cVoxelNeighborsIndexAdd[i];
                auto * pSet = sepStorage.findSeparationPointSet( index );
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
                        // `neis` indicates that current voxel has valid point for desired triangulation
                        // as far as nei has 3 directions we use `dir` to validate (make sure that there is point in needed edge) desired direction
                        voxelValid = voxelValid && neis[interIndex0] && (*neis[interIndex0])[int( dir0 )];
                        voxelValid = voxelValid && neis[interIndex1] && (*neis[interIndex1])[int( dir1 )];
                        voxelValid = voxelValid && neis[interIndex2] && (*neis[interIndex2])[int( dir2 )];
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
                assert( neis[interIndex0] && (*neis[interIndex0])[int( dir0 )] );
                assert( neis[interIndex1] && (*neis[interIndex1])[int( dir1 )] );
                assert( neis[interIndex2] && (*neis[interIndex2])[int( dir2 )] );

                if ( params.lessInside )
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
                if ( params.outVoxelPerFaceMap )
                    block.faceMap.emplace_back( VoxelId{ ind } );
            }

            if ( runCallback && ( ind - begin ) % 16384 == 0 )
                if ( !subprogress2( float( ind - begin ) / float( end - begin ) ) )
                    keepGoing.store( false, std::memory_order_relaxed );
        }
    } );

    if ( params.cb && !keepGoing )
        return unexpectedOperationCanceled();

    // create result triangulation
    result.tris = sepStorage.getTriangulation( params.outVoxelPerFaceMap );

    if ( params.cb && !params.cb( 0.95f ) )
        return unexpectedOperationCanceled();

    // some points may be not referenced by any triangle due to NaNs
    result.points.resize( totalVertices );
    sepStorage.getPoints( result.points );

    if ( params.cb && !params.cb( 1.0f ) )
        return unexpectedOperationCanceled();

    return result;
}

template <typename V, typename NaNChecker>
Expected<TriMesh> volumeToMeshHelper1( const V& volume, const MarchingCubesParams& params, NaNChecker&& nanChecker )
{
    if ( params.positioner )
        return volumeToMesh( volume, params, std::forward<NaNChecker>( nanChecker ), params.positioner );

    return volumeToMesh( volume, params, std::forward<NaNChecker>( nanChecker ),
        []( const Vector3f& pos0, const Vector3f& pos1, float v0, float v1, float iso )
        {
            assert( v0 != v1 );
            const auto ratio = ( iso - v0 ) / ( v1 - v0 );
            assert( ratio >= 0 && ratio <= 1 );
            return ( 1.0f - ratio ) * pos0 + ratio * pos1;
        } );
}

template <typename V>
Expected<TriMesh> volumeToMeshHelper2( const V& volume, const MarchingCubesParams& params )
{
    if ( params.omitNaNCheck )
        return volumeToMeshHelper1( volume, params, [] ( float ) { return false; } );
    else
        return volumeToMeshHelper1( volume, params, isNanFast );
}

Expected<TriMesh> marchingCubesAsTriMesh( const SimpleVolume& volume, const MarchingCubesParams& params /*= {} */ )
{
    if ( params.iso <= volume.min || params.iso >= volume.max )
        return TriMesh{};
    return volumeToMeshHelper2( volume, params );
}

Expected<Mesh> marchingCubes( const SimpleVolume& volume, const MarchingCubesParams& params )
{
    MR_TIMER
    auto p = params;
    p.cb = subprogress( params.cb, 0.0f, 0.9f );
    return marchingCubesAsTriMesh( volume, p ).and_then( [&params]( TriMesh && tm ) -> Expected<Mesh>
    {
        return Mesh::fromTriMesh( std::move( tm ), {}, subprogress( params.cb, 0.9f, 1.0f ) );
    } );
}

#ifndef MRMESH_NO_OPENVDB
Expected<TriMesh> marchingCubesAsTriMesh( const VdbVolume& volume, const MarchingCubesParams& params /*= {} */ )
{
    if ( !volume.data )
        return unexpected( "No volume data." );
    if ( params.iso <= volume.min || params.iso >= volume.max )
        return TriMesh{};
    return volumeToMeshHelper2( volume, params );
}

Expected<Mesh> marchingCubes( const VdbVolume& volume, const MarchingCubesParams& params /*= {} */ )
{
    MR_TIMER
    auto p = params;
    p.cb = subprogress( params.cb, 0.0f, 0.9f );
    return marchingCubesAsTriMesh( volume, p ).and_then( [&params]( TriMesh && tm ) -> Expected<Mesh>
    {
        return Mesh::fromTriMesh( std::move( tm ), {}, subprogress( params.cb, 0.9f, 1.0f ) );
    } );
}
#endif

Expected<TriMesh> marchingCubesAsTriMesh( const FunctionVolume& volume, const MarchingCubesParams& params )
{
    if ( !volume.data )
        return unexpected( "Getter function is not specified." );
    return volumeToMeshHelper2( volume, params );
}

Expected<Mesh> marchingCubes( const FunctionVolume& volume, const MarchingCubesParams& params )
{
    MR_TIMER
    auto p = params;
    p.cb = subprogress( params.cb, 0.0f, 0.9f );
    return marchingCubesAsTriMesh( volume, p ).and_then( [&params]( TriMesh && tm ) -> Expected<Mesh>
    {
        return Mesh::fromTriMesh( std::move( tm ), {}, subprogress( params.cb, 0.9f, 1.0f ) );
    } );
}

} //namespace MR
