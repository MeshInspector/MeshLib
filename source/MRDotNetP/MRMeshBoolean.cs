using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    /// Available CSG operations
    enum BooleanOperation
    {
        /// Part of mesh `A` that is inside of mesh `B`
        InsideA,
        /// Part of mesh `B` that is inside of mesh `A`
        InsideB,
        /// Part of mesh `A` that is outside of mesh `B`
        OutsideA,
        /// Part of mesh `B` that is outside of mesh `A`
        OutsideB,
        /// Union surface of two meshes (outside parts)
        Union,
        /// Intersection surface of two meshes (inside parts)
        Intersection,
        /// Surface of mesh `B` - surface of mesh `A` (outside `B` - inside `A`)
        DifferenceBA,
        /// Surface of mesh `A` - surface of mesh `B` (outside `A` - inside `B`)
        DifferenceAB
    };

/// Input object index enum


    class MeshBoolean
    {
        /// creates a new BooleanResultMapper object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperNew();

        /// Returns faces bitset of result mesh corresponding input one
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperMapFaces( IntPtr mapper, IntPtr oldBS, BooleanResultMapperMapObject obj );

        /// Returns vertices bitset of result mesh corresponding input one
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperMapVerts( IntPtr mapper, IntPtr oldBS, BooleanResultMapperMapObject obj );

        /// Returns edges bitset of result mesh corresponding input one
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MREdgeBitSet* mrBooleanResultMapperMapEdges( const MRBooleanResultMapper* mapper, const MREdgeBitSet* oldBS, MRBooleanResultMapperMapObject obj );

/// Returns only new faces that are created during boolean operation
MRMESHC_API MRFaceBitSet* mrBooleanResultMapperNewFaces( const MRBooleanResultMapper* mapper );

        /// returns updated oldBS leaving only faces that has corresponding ones in result mesh
        MRMESHC_API MRFaceBitSet* mrBooleanResultMapperFilteredOldFaceBitSet(MRBooleanResultMapper* mapper, const MRFaceBitSet* oldBS, MRBooleanResultMapperMapObject obj );

MRMESHC_API const MRBooleanResultMapperMaps* mrBooleanResultMapperGetMaps( const MRBooleanResultMapper* mapper, MRBooleanResultMapperMapObject index );

/// "after cut" faces to "origin" faces
/// this map is not 1-1, but N-1
MRMESHC_API const MRFaceMap mrBooleanResultMapperMapsCut2origin( const MRBooleanResultMapperMaps* maps );

        /// "after cut" faces to "after stitch" faces (1-1)
        MRMESHC_API const MRFaceMap mrBooleanResultMapperMapsCut2newFaces( const MRBooleanResultMapperMaps* maps );

        /// "origin" edges to "after stitch" edges (1-1)
        MRMESHC_API const MRWholeEdgeMap mrBooleanResultMapperMapsOld2newEdges( const MRBooleanResultMapperMaps* maps );

        /// "origin" vertices to "after stitch" vertices (1-1)
        MRMESHC_API const MRVertMap mrBooleanResultMapperMapsOld2NewVerts( const MRBooleanResultMapperMaps* maps );

        /// old topology indexes are valid if true
        MRMESHC_API bool mrBooleanResultMapperMapsIdentity( const MRBooleanResultMapperMaps* maps );

        /// deallocates a BooleanResultMapper object
        MRMESHC_API void mrBooleanResultMapperFree(MRBooleanResultMapper* mapper);
    }
}
