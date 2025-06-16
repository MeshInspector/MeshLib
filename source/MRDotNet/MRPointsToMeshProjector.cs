using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using static MR.DotNet;

namespace MR
{
    using VertScalars = List<float>;
    public partial class DotNet
    {

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRMeshProjectionParameters
        {
            public float loDistLimitSq=0;
            public float upDistLimitSq = float.MaxValue;
            public IntPtr refXf = IntPtr.Zero;
            public IntPtr xf = IntPtr.Zero;
            public MRMeshProjectionParameters() { }
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRScalars
        {
            public IntPtr data = IntPtr.Zero;
            public ulong size = 0;
            public IntPtr reserved = IntPtr.Zero;
            public MRScalars() { }
        }

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        unsafe private static extern MRScalars* mrFindSignedDistances(IntPtr meshA, IntPtr meshB, ref MRMeshProjectionParameters parameters);

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        unsafe private static extern void mrScalarsFree(MRScalars* vector);

        public struct MeshProjectionParameters
        {
            /// minimum squared distance from a test point to mesh to be computed precisely,
            /// if a mesh point is found within this distance then it is immediately returned without searching for a closer one
            public float loDistLimitSq = 0;

            /// maximum squared distance from a test point to mesh to be computed precisely,
            /// if actual distance is larger than upDistLimit will be returned with not-trusted sign
            public float upDistLimitSq = float.MaxValue;

            /// optional reference mesh to world transformation
            public AffineXf3f? refXf = null;

            /// optional test points to world transformation
            public AffineXf3f? xf = null;

            public MeshProjectionParameters() { }
        };

        /// Computes signed distances from valid vertices of test mesh to the closest point on the reference mesh:
        /// positive value - outside reference mesh, negative - inside reference mesh;
        /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
        unsafe public static VertScalars FindSignedDistances(Mesh refMesh,Mesh mesh, MeshProjectionParameters parameters)
        {
            MRMeshProjectionParameters innerParams;
            innerParams.loDistLimitSq = parameters.loDistLimitSq;
            innerParams.upDistLimitSq = parameters.upDistLimitSq;
            innerParams.refXf = parameters.refXf is null ? (IntPtr)null : parameters.refXf.XfAddr();
            innerParams.xf = parameters.xf is null ? (IntPtr)null : parameters.xf.XfAddr();

            var res = mrFindSignedDistances(refMesh.mesh_, mesh.mesh_, ref innerParams);

            var scalars = new VertScalars();
            if (res == null)
                return scalars;
            var mrVertScalars = *res;
            int sizeOfFloat = Marshal.SizeOf(typeof(float));
            var mrVertScalarsData = mrVertScalars.data;

            // copy data
            for (int i = 0; i < (int)mrVertScalars.size; i++)
            {
                var flt = Marshal.PtrToStructure<float>(IntPtr.Add(mrVertScalarsData, i * sizeOfFloat));
                scalars.Add(flt);
            }
            mrScalarsFree(res);
            return scalars;
        }
    }
}