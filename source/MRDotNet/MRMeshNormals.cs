using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using System.Text;
using static MR.DotNet;

namespace MR
{
    using VertNormals = List<Vector3f>;
    using FaceNormals = List<Vector3f>;

    public partial class DotNet
    {

        public class MeshNormals
        {
            [StructLayout(LayoutKind.Sequential)]
            internal struct MRFaceNormals
            {
                public IntPtr data = IntPtr.Zero;
                public ulong size = 0;
                public IntPtr reserved = IntPtr.Zero;
                public MRFaceNormals() { }
            }

            /// returns a vector with face-normal in every element for valid mesh faces
            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            unsafe private static extern MRFaceNormals* mrComputePerFaceNormals(IntPtr mesh);

            /// returns a vector with vertex normals in every element for valid mesh vertices
            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            unsafe private static extern MRFaceNormals* mrComputePerVertNormals(IntPtr mesh);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            unsafe private static extern void mrVectorVector3fFree(MRFaceNormals* p);

            unsafe public static VertNormals ComputePerVertNormals(Mesh mesh)
            {
                var p = mrComputePerVertNormals(mesh.mesh_);

                var normals = new VertNormals();
                if (p == null)
                    return normals;
                var mrVertNormals = *p;
                int sizeOfMRVector3f = Marshal.SizeOf(typeof(Vector3f.MRVector3f));
                var mrVertNormalsData = mrVertNormals.data;

                // copy data
                for (int i = 0; i < (int)mrVertNormals.size; i++)
                {
                    var mrVector3f = Marshal.PtrToStructure<Vector3f.MRVector3f>(IntPtr.Add(mrVertNormalsData, i * sizeOfMRVector3f));
                    normals.Add(new Vector3f(mrVector3f));
                }
                mrVectorVector3fFree(p);
                return normals;
            }
            /// returns a list with face normals in every element for valid mesh faces
            unsafe public static FaceNormals ComputePerFaceNormals(Mesh mesh)
            {
                var p = mrComputePerFaceNormals(mesh.mesh_);

                var normals = new FaceNormals();
                if (p == null)
                    return normals;
                var mrFaceNormals = *p;
                int sizeOfMRVector3f = Marshal.SizeOf(typeof(Vector3f.MRVector3f));
                var mrFaceNormalsData = mrFaceNormals.data;

                // copy data
                for (int i = 0; i < (int)mrFaceNormals.size; i++)
                {
                    var mrVector3f = Marshal.PtrToStructure<Vector3f.MRVector3f>(IntPtr.Add(mrFaceNormalsData, i * sizeOfMRVector3f));
                    normals.Add(new Vector3f(mrVector3f));
                }
                mrVectorVector3fFree(p);
                return normals;
            }
        }
    }
}

