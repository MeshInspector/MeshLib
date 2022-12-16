#pragma once

#include "MRMeshFwd.h"

#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types
#pragma clang diagnostic push
#pragma GCC diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <Eigen/Core>
#pragma GCC diagnostic pop
#pragma clang diagnostic pop
#pragma warning(pop)

namespace MR
{

/// \defgroup MeshEigenGroup Mesh Eigen
/// \ingroup MeshAlgorithmGroup
/// \{

/// constructs mesh topology from N*3 matrix of vertex indices
MRMESH_API MeshTopology topologyFromEigen( const Eigen::MatrixXi & F );

/// constructs mesh from M*3 matrix of coordinates and N*3 matrix of vertex indices
MRMESH_API Mesh meshFromEigen( const Eigen::MatrixXd & V, const Eigen::MatrixXi & F );

/// replace selected points with the values from V
MRMESH_API void pointsFromEigen( const Eigen::MatrixXd & V, const VertBitSet & selection, VertCoords & points );

/// converts valid faces from mesh topology into N*3 matrix of vertex indices
MRMESH_API void topologyToEigen( const MeshTopology & topology, Eigen::MatrixXi & F );

/// converts mesh into M*3 matrix of coordinates and N*3 matrix of vertex indices
MRMESH_API void meshToEigen( const Mesh & mesh, Eigen::MatrixXd & V, Eigen::MatrixXi & F );

/// \}

} // namespace MR
