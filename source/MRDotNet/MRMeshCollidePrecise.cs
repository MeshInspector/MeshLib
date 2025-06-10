using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using static MR.DotNet;

namespace MR
{
    using PreciseCollisionResult = VectorVarEdgeTri;

    public partial class DotNet
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct EdgeTri
        {
            EdgeId edge;
            FaceId tri;
        };

        [StructLayout(LayoutKind.Explicit, Size = 4)]
        public class FlaggedTri
        {
            [FieldOffset(0)] private byte byte0;
            [FieldOffset(0)] private UInt32 data32;

            public bool isEdgeATriB
            {
                get { return Convert.ToBoolean(byte0 >> 7); }
                set { byte0 = (byte)((byte0 & 0x7f) + (value ? 0x80 : 0x00)); }
            }

            public uint face
            {
                get { return data32 & 0x7fffffff; }
                set { data32 = (data32 & 0x80000000) + value; }
            }

            FlaggedTri(bool isEdgeATriB_, int face_)
            {
                Debug.Assert(face_ >= 0, "face id must be valid");
                isEdgeATriB = isEdgeATriB_;
                face = (uint)face_;
            }
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct VarEdgeTri
        {
            EdgeId edge;
            FlaggedTri flaggedTri;

            public bool isEdgeATriB
            {
                get { return flaggedTri.isEdgeATriB; }
            }

            public uint face
            {
                get { return flaggedTri.face; }
            }
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRVectorVarEdgeTri
        {
            public IntPtr data = IntPtr.Zero;
            public ulong size = 0;
            public IntPtr reserved = IntPtr.Zero;
            public MRVectorVarEdgeTri() { }
        }

        public class VectorVarEdgeTri : IDisposable
        {
            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern void mrVectorVarEdgeTriFree(IntPtr vector);

            internal VectorVarEdgeTri(IntPtr native)
            {
                nativeVector_ = native;
                vector_ = Marshal.PtrToStructure<MRVectorVarEdgeTri>(native);
            }

            internal VectorVarEdgeTri(MRVectorVarEdgeTri vector)
            {
                vector_ = vector;
                nativeVector_ = IntPtr.Zero;
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
                    if (nativeVector_ != IntPtr.Zero)
                    {
                        mrVectorVarEdgeTriFree(nativeVector_);
                        nativeVector_ = IntPtr.Zero;
                    }

                    disposed = true;
                }
            }

            ~VectorVarEdgeTri()
            {
                Dispose(false);
            }

            public ReadOnlyCollection<VarEdgeTri> List
            {
                get
                {
                    if (list_ is null)
                    {
                        list_ = new List<VarEdgeTri>((int)vector_.size);

                        var vetSize = Marshal.SizeOf(typeof(VarEdgeTri));
                        for (int i = 0; i < (int)vector_.size; i++)
                        {
                            var vetPtr = IntPtr.Add(vector_.data, i * vetSize);
                            var vet = Marshal.PtrToStructure<VarEdgeTri>(vetPtr);
                            list_.Add(vet);
                        }
                    }

                    return list_.AsReadOnly();
                }
            }

            private List<VarEdgeTri>? list_ = null;

            private MRVectorVarEdgeTri vector_;
            internal IntPtr nativeVector_;
        }

        /**
         * \brief finds all pairs of colliding edges from one mesh and triangle from another mesh
         * \param rigidB2A rigid transformation from B-mesh space to A mesh space, NULL considered as identity transformation
         * \param anyIntersection if true then the function returns as fast as it finds any intersection
         */
        [DllImport("MRMeshC", CharSet = CharSet.Auto)]
        private static extern IntPtr mrFindCollidingEdgeTrisPrecise(ref MRMeshPart a, ref MRMeshPart b, IntPtr conv, IntPtr rigidB2A, bool anyIntersection);

        public static PreciseCollisionResult FindCollidingEdgeTrisPrecise(MeshPart meshA, MeshPart meshB, CoordinateConverters conv, AffineXf3f? rigidB2A = null, bool anyIntersection = false)
        {
            var mrResult = mrFindCollidingEdgeTrisPrecise(ref meshA.mrMeshPart, ref meshB.mrMeshPart, conv.GetConvertToIntVector(), rigidB2A is null ? IntPtr.Zero : rigidB2A.XfAddr(), anyIntersection);

            return new PreciseCollisionResult(mrResult);
        }
    }
}
