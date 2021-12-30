#pragma once
#include "exports.h"
#include "MRMesh/MRVector.h"
#include <tl/expected.hpp>
#include <array>

namespace MRE
{

/** \ingroup BooleanGroup
  * Enum class of available CSG operations
  * \image html boolean/no_bool.png "Two separate meshes" width = 300cm
  * \sa \ref MRE::boolean
  */
enum class BooleanOperation
{
    /// Part of mesh `A` that is inside of mesh `B`
    /// \image html boolean/inside_a.png "Inside A" width = 300cm
    InsideA,
    /// Part of mesh `B` that is inside of mesh `A`
    /// \image html boolean/inside_b.png "Inside B" width = 300cm
    InsideB,
    /// Part of mesh `A` that is outside of mesh `B`
    /// \image html boolean/outside_a.png "Outside A" width = 300cm
    OutsideA,
    /// Part of mesh `B` that is outside of mesh `A`
    /// \image html boolean/outside_b.png "Outside B" width = 300cm
    OutsideB,
    /// Union surface of two meshes (outside parts)
    /// \image html boolean/union.png "Union" width = 300cm
    Union,
    /// Intersection surface of two meshes (inside parts)
    /// \image html boolean/intersection.png "Intersection" width = 300cm
    Intersection,
    /// Surface of mesh `B` - surface of mesh `A` (outside `B` - inside `A`)
    /// \image html boolean/b-a.png "Difference B-A" width = 300cm
    DifferenceBA,
    /// Surface of mesh `A` - surface of mesh `B` (outside `A` - inside `B`)
    /// \image html boolean/a-b.png "Difference A-B" width = 300cm
    DifferenceAB,

    Count ///< not a valid operation
};

/// Structure to map old mesh BitSets to new
/** \struct MRE::BooleanResultMapper
  * \ingroup BooleanGroup
  * \brief Structure to easily map topology of MRE::boolean input meshes to result mesh
  * 
  * This structure allows to map faces, vertices and edges of mesh `A` and mesh `B` input of MRE::boolean to result mesh topology primitives
  * \sa \ref MRE::boolean
  */
struct BooleanResultMapper
{
    /// Input object index enum
    enum class MapObject { A, B, Count };

    BooleanResultMapper() = default;
    
    /// Returns faces bitset of result mesh corresponding input one
    MREALGORITHMS_API MR::FaceBitSet map( const MR::FaceBitSet& oldBS, MapObject obj ) const;
    /// Returns vertices bitset of result mesh corresponding input one
    MREALGORITHMS_API MR::VertBitSet map( const MR::VertBitSet& oldBS, MapObject obj ) const;
    /// Returns edges bitset of result mesh corresponding input one
    MREALGORITHMS_API MR::EdgeBitSet map( const MR::EdgeBitSet& oldBS, MapObject obj ) const;

    struct Maps
    {
        /// "after cut" faces to "origin" faces
        /// this map is not 1-1, but N-1
        MR::FaceMap cut2origin;
        /// "after cut" faces to "after stitch" faces (1-1)
        MR::FaceMap cut2newFaces;
        /// "origin" edges to "after stitch" edges (1-1)
        MR::EdgeMap old2newEdges;
        /// "origin" vertices to "after stitch" vertices (1-1)
        MR::VertMap old2newVerts;
        /// old topology indexes are valid if true
        bool identity{false};
    };
    std::array<Maps, size_t( MapObject::Count )> maps;
};


// Perform boolean operation on cut meshes,
// returns mesh in space of meshA or error.
// note: actually this function is meant to be internal, use "boolean" instead
MREALGORITHMS_API tl::expected<MR::Mesh, std::string> doBooleanOperation( const MR::Mesh& meshACut, const MR::Mesh& meshBCut,
                                                               const std::vector<MR::EdgePath>& cutEdgesA, const std::vector<MR::EdgePath>& cutEdgesB,
                                                               BooleanOperation operation, const MR::AffineXf3f* rigidB2A = nullptr,
                                                               BooleanResultMapper* mapper = nullptr );

}