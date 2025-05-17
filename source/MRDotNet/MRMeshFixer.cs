using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR
{
    public partial class DotNet
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct MultipleEdge
        {
            VertId v0;
            VertId v1;
        }
        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFindHoleComplicatingFaces(IntPtr mesh);

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFindDegenerateFaces(ref MRMeshPart mp, float criticalAspectRatio, IntPtr cb, ref IntPtr errorStr);

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFindShortEdges(ref MRMeshPart mp, float criticalLength, IntPtr cb, ref IntPtr errorStr);

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        unsafe private static extern void fixMultipleEdges(IntPtr mesh, MultipleEdge* multipleEdges, ulong multipleEdgesNum);

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        private static extern void findAndFixMultipleEdges(IntPtr mesh);

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrStringData(IntPtr str);

        /// returns all faces that complicate one of mesh holes;
        /// hole is complicated if it passes via one vertex more than once;
        /// deleting such faces simplifies the holes and makes them easier to fill
        public static BitSet FindHoleComplicatingFaces(Mesh mesh)
        {
            return new BitSet(mrFindHoleComplicatingFaces(mesh.mesh_));
        }
        /// finds faces having aspect ratio >= criticalAspectRatio
        public static FaceBitSet FindDegenerateFaces(MeshPart meshPart, float criticalAspectRatio)
        {
            IntPtr errorString = new IntPtr();
            IntPtr res = mrFindDegenerateFaces(ref meshPart.mrMeshPart, criticalAspectRatio, IntPtr.Zero, ref errorString);
            if (errorString != IntPtr.Zero)
            {
                var errData = mrStringData(errorString);
                string errorMessage = MarshalNativeUtf8ToManagedString(errData);
                throw new Exception(errorMessage);
            }
            return new FaceBitSet(res);
        }
        /// finds edges having length <= criticalLength
        public static UndirectedEdgeBitSet FindShortEdges(MeshPart meshPart, float criticalLength)
        {
            IntPtr errorString = new IntPtr();
            IntPtr res = mrFindShortEdges(ref meshPart.mrMeshPart, criticalLength, IntPtr.Zero, ref errorString);
            if (errorString != IntPtr.Zero)
            {
                var errData = mrStringData(errorString);
                string errorMessage = MarshalNativeUtf8ToManagedString(errData);
                throw new Exception(errorMessage);
            }
            return new UndirectedEdgeBitSet(res);
        }
        /// resolves given multiple edges, but splitting all but one edge in each group
        unsafe public static void FixMultipleEdges(ref Mesh mesh, List<MultipleEdge> multipleEdges)
        {
            MultipleEdge* multipleEdgesPtr = stackalloc MultipleEdge[multipleEdges.Count];
            for (int i = 0; i < multipleEdges.Count; i++)
                multipleEdgesPtr[i] = multipleEdges[i];

            fixMultipleEdges(mesh.mesh_, multipleEdgesPtr, (ulong)multipleEdges.Count);
        }
        /// finds and resolves multiple edges
        public static void FixMultipleEdges(ref Mesh mesh)
        {
            findAndFixMultipleEdges(mesh.mesh_);
        }

        public struct FixMeshDegeneraciesParams
        {
            /// maximum permitted deviation from the original surface
            public float maxDeviation = 0f;

            /// edges not longer than this value will be collapsed ignoring normals and aspect ratio checks
            public float tinyEdgeLength = 0f;

            /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
            /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
            public float criticalTriAspectRatio = 1e4f;

            /// Permit edge flips if it does not change dihedral angle more than on this value
            public float maxAngleChange = (float)(Math.PI / 3.0);

            /// Small stabilizer is important to achieve good results on completely planar mesh parts,
            /// if your mesh is not-planer everywhere, then you can set it to zero
            public float stabilizer = 1e-6f;

            /// degenerations will be fixed only in given region, it is updated during the operation
            public FaceBitSet? region = null;

            public enum Mode
            {
                Decimate, ///< use decimation only to fix degeneracies
                Remesh,   ///< if decimation does not succeed, perform subdivision too
                RemeshPatch ///< if both decimation and subdivision does not succeed, removes degenerate areas and fills occurred holes
            }
            public Mode mode = Mode.Remesh;

            public FixMeshDegeneraciesParams() { }
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRFixMeshDegeneraciesParams
        {
            public float maxDeviation = 0f;
            public float tinyEdgeLength = 0f;
            public float criticalTriAspectRatio = 1e4f;
            public float maxAngleChange = (float)(Math.PI / 3.0);
            public float stabilizer = 1e-6f;
            public IntPtr region = IntPtr.Zero;
            public FixMeshDegeneraciesParams.Mode mode = FixMeshDegeneraciesParams.Mode.Remesh;
            public IntPtr cb = IntPtr.Zero;

            public MRFixMeshDegeneraciesParams() { }
        }

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        private static extern void mrFixMeshDegeneracies(IntPtr mesh, ref MRFixMeshDegeneraciesParams settings, ref IntPtr errorString);

        /// Fixes degenerate faces and short edges in mesh (changes topology)
        public static void FixMeshDegeneracies(ref Mesh mesh, FixMeshDegeneraciesParams settings)
        {
            MRFixMeshDegeneraciesParams mrParams;
            mrParams.maxDeviation = settings.maxDeviation;
            mrParams.tinyEdgeLength = settings.tinyEdgeLength;
            mrParams.criticalTriAspectRatio = settings.criticalTriAspectRatio;
            mrParams.maxAngleChange = settings.maxAngleChange;
            mrParams.stabilizer = settings.stabilizer;
            mrParams.region = settings.region?.bs_ ?? IntPtr.Zero;
            mrParams.mode = settings.mode;
            mrParams.cb = IntPtr.Zero;

            IntPtr errorString = IntPtr.Zero;
            mrFixMeshDegeneracies(mesh.varMesh(), ref mrParams, ref errorString);
            if ( errorString != IntPtr.Zero )
            {
                string error = MarshalNativeUtf8ToManagedString( mrStringData( errorString ) );
                throw new SystemException( error );
            }
        }
    }
}
