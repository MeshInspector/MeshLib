using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using System.Text;

namespace MR
{
    public partial class DotNet
    {

        [StructLayout(LayoutKind.Sequential)]
        public struct EdgeTri
        {
            EdgeId edge;
            FaceId tri;
        };
        public class PreciseCollisionResult : IDisposable
        {
            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern bool mrEdgeTriEq(ref EdgeTri a, ref EdgeTri b);

            [StructLayout(LayoutKind.Sequential)]
            internal struct MRVectorEdgeTri
            {
                public IntPtr data = IntPtr.Zero;
                public ulong size = 0;
                public IntPtr reserved = IntPtr.Zero;
                public MRVectorEdgeTri() { }
            };

            /// each edge is directed to have its origin inside and its destination outside of the other mesh
            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVectorEdgeTri mrPreciseCollisionResultEdgesAtrisB(IntPtr result);

            /// each edge is directed to have its origin inside and its destination outside of the other mesh
            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVectorEdgeTri mrPreciseCollisionResultEdgesBtrisA(IntPtr result);

            /// deallocates the PreciseCollisionResult object
            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern void mrPreciseCollisionResultFree(IntPtr result);



            internal PreciseCollisionResult(IntPtr nativeResult)
            {
                nativeResult_ = nativeResult;
            }

            private bool disposed = false;
            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            protected virtual void Dispose(bool disposing)
            {
                if (!disposed)
                {
                    if (nativeResult_ != IntPtr.Zero)
                    {
                        mrPreciseCollisionResultFree(nativeResult_);
                    }

                    disposed = true;
                }
            }
            ~PreciseCollisionResult()
            {
                Dispose(false);
            }

            /// each edge is directed to have its origin inside and its destination outside of the other mesh
            public ReadOnlyCollection<EdgeTri> EdgesAtrisB
            {
                get
                {
                    if (edgesAtrisB_ is null)
                    {
                        var mrEdges = mrPreciseCollisionResultEdgesAtrisB(nativeResult_);
                        int sizeOfEdgeTri = Marshal.SizeOf(typeof(EdgeTri));
                        edgesAtrisB_ = new List<EdgeTri>((int)mrEdges.size);
                        for (int i = 0; i < (int)mrEdges.size; ++i)
                        {
                            edgesAtrisB_.Add(Marshal.PtrToStructure<EdgeTri>(IntPtr.Add(mrEdges.data, i * sizeOfEdgeTri)));
                        }
                    }
                    return edgesAtrisB_.AsReadOnly();
                }
            }
            /// each edge is directed to have its origin inside and its destination outside of the other mesh
            public ReadOnlyCollection<EdgeTri> EdgesBtrisA
            {
                get
                {
                    if (edgesBtrisA_ is null)
                    {
                        var mrEdges = mrPreciseCollisionResultEdgesBtrisA(nativeResult_);
                        int sizeOfEdgeTri = Marshal.SizeOf(typeof(EdgeTri));
                        edgesBtrisA_ = new List<EdgeTri>((int)mrEdges.size);
                        for (int i = 0; i < (int)mrEdges.size; ++i)
                        {
                            edgesBtrisA_.Add(Marshal.PtrToStructure<EdgeTri>(IntPtr.Add(mrEdges.data, i * sizeOfEdgeTri)));
                        }
                    }
                    return edgesBtrisA_.AsReadOnly();
                }
            }

            private List<EdgeTri>? edgesAtrisB_;
            private List<EdgeTri>? edgesBtrisA_;

            internal IntPtr nativeResult_;
        };
        /**
         * \brief finds all pairs of colliding edges from one mesh and triangle from another mesh
         * \param rigidB2A rigid transformation from B-mesh space to A mesh space, NULL considered as identity transformation
         * \param anyIntersection if true then the function returns as fast as it finds any intersection
         */
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrFindCollidingEdgeTrisPrecise(ref MRMeshPart a, ref MRMeshPart b, IntPtr conv, IntPtr rigidB2A, bool anyIntersection);

        public static PreciseCollisionResult FindCollidingEdgeTrisPrecise(MeshPart meshA, MeshPart meshB, CoordinateConverters conv, AffineXf3f? rigidB2A = null, bool anyIntersection = false)
        {
            var mrResult = mrFindCollidingEdgeTrisPrecise(ref meshA.mrMeshPart, ref meshB.mrMeshPart, conv.GetConvertToIntVector(), rigidB2A is null ? IntPtr.Zero : rigidB2A.XfAddr(), anyIntersection);

            return new PreciseCollisionResult(mrResult);
        }
    }
}
