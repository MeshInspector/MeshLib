using System;
using System.Data.Common;
using System.Runtime.InteropServices;
using static MR.DotNet.Vector3f;
//using static MR.DotNet.Vector3f;

namespace MR.DotNet
{
    public class Mesh
    {
        private IntPtr mesh_;
        private IntPtr meshTopology_;

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRVertId
        {
            public int id;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MREdgeId
        {
            public int id;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRFaceId
        {
            public int id;
        }


        /// tightly packs all arrays eliminating lone edges and invalid faces and vertices
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrMeshTopologyPack(IntPtr top);

        /// returns cached set of all valid vertices
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrMeshTopologyGetValidVerts( IntPtr top );

        /// returns cached set of all valid faces
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrMeshTopologyGetValidFaces( IntPtr top );


        /// returns three vertex ids for valid triangles (which can be accessed by FaceId),
        /// vertex ids for invalid triangles are undefined, and shall not be read
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrMeshTopologyGetTriangulation( IntPtr top );

        /// returns the number of face records including invalid ones
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrMeshTopologyFaceSize( IntPtr top );
        /// returns one edge with no valid left face for every boundary in the mesh
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrMeshTopologyFindHoleRepresentiveEdges( IntPtr top );

        /// gets 3 vertices of given triangular face;
        /// the vertices are returned in counter-clockwise order if look from mesh outside
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrMeshTopologyGetLeftTriVerts( IntPtr top, MREdgeId a, ref MRVertId v0, ref MRVertId v1, ref MRVertId v2 );

        /// returns the number of hole loops in the mesh;
        /// \param holeRepresentativeEdges optional output of the smallest edge id with no valid left face in every hole
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern int mrMeshTopologyFindNumHoles( IntPtr top, IntPtr holeRepresentativeEdges );

        MRMESHC_API MRMesh* mrMeshFromTriangles( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

/// constructs a mesh from vertex coordinates and a set of triangles with given ids;
/// unlike simple \ref mrMeshFromTriangles it tries to resolve non-manifold vertices by creating duplicate vertices
MRMESHC_API MRMesh* mrMeshFromTrianglesDuplicatingNonManifoldVertices( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

            MRMESHC_API const MRVector3f* mrMeshPoints( const MRMesh* mesh );

        MRMESHC_API size_t mrMeshPointsNum( const MRMesh* mesh );

    }
}
