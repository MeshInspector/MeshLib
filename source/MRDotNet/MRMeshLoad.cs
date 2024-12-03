using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using static MR.DotNet.AffineXf3f;

namespace MR
{
    public partial class DotNet
    {
        public struct ObjLoadSettings
        {
            /// if true then vertices will be returned relative to some transformation to avoid precision loss
            [MarshalAs(UnmanagedType.U1)]
            public bool customXf = false;
            /// if true, the number of skipped faces (faces than can't be created) will be counted
            [MarshalAs(UnmanagedType.U1)]
            public bool countSkippedFaces = false;
            public ObjLoadSettings() { }
        };

        public struct NamedMesh
        {
            public string name = "";
            public Mesh? mesh = null;
            /// transform of the loaded mesh, not identity only if ObjLoadSettings.customXf
            public AffineXf3f? xf = null;
            /// counter of skipped faces (faces than can't be created), not zero only if ObjLoadSettings.countSkippedFaces
            public int skippedFaceCount = 0;
            /// counter of duplicated vertices (that created for resolve non-manifold geometry)
            public int duplicatedVertexCount = 0;

            public NamedMesh() { }
        };

        // inherits List<NamedMesh> and correctly disposes native resource
        public class NamedMeshList : List<NamedMesh>, IDisposable
        {
            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern void mrVectorMeshLoadNamedMeshFree(IntPtr vector);

            internal NamedMeshList(IntPtr nativeList) : base()
            { nativeList_ = nativeList; }

            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            protected virtual void Dispose(bool disposing)
            {
                if (!disposed_)
                {
                    if (nativeList_ != IntPtr.Zero)
                    {
                        mrVectorMeshLoadNamedMeshFree(nativeList_);
                        nativeList_ = IntPtr.Zero;
                    }

                    disposed_ = true;
                }
            }

            ~NamedMeshList()
            {
                Dispose(false);
            }

            private IntPtr nativeList_;
            private bool disposed_ = false;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRMeshLoadObjLoadSettings
        {
            /// if true then vertices will be returned relative to some transformation to avoid precision loss
            public bool customXf = false;
            /// if true, the number of skipped faces (faces than can't be created) will be counted
            public bool countSkippedFaces = false;
            /// callback for set progress and stop process
            public IntPtr callback = IntPtr.Zero;

            public MRMeshLoadObjLoadSettings() { }
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRMeshLoadNamedMesh
        {
            public IntPtr name = IntPtr.Zero;
            public IntPtr mesh = IntPtr.Zero;
            /// transform of the loaded mesh, not identity only if ObjLoadSettings.customXf
            public MRAffineXf3f xf = new MRAffineXf3f();
            /// counter of skipped faces (faces than can't be created), not zero only if ObjLoadSettings.countSkippedFaces
            public int skippedFaceCount = 0;
            /// counter of duplicated vertices (that created for resolve non-manifold geometry)
            public int duplicatedVertexCount = 0;
            public MRMeshLoadNamedMesh() { }
        };

        public class MeshLoad
        {
            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern MRMeshLoadNamedMesh mrVectorMeshLoadNamedMeshGet(IntPtr vector, ulong index);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern ulong mrVectorMeshLoadNamedMeshSize(IntPtr vector);




            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrMeshLoadFromSceneObjFile(string file, bool combineAllObjects, ref MRMeshLoadObjLoadSettings settings, ref IntPtr errorString);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrStringData(IntPtr str);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            unsafe private static extern IntPtr mrMeshLoadFromAnySupportedFormat(string file, IntPtr* errorStr);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern void mrLoadIOExtras();
            /// loads mesh from file of any supported format
            unsafe public static Mesh FromAnySupportedFormat(string path)
            {
                mrLoadIOExtras();

                IntPtr errString = new IntPtr();
                var mesh = mrMeshLoadFromAnySupportedFormat(path, &errString);

                if (errString != IntPtr.Zero)
                {
                    var errData = mrStringData(errString);
                    string errorMessage = Marshal.PtrToStringAnsi(errData);
                    throw new SystemException(errorMessage);
                }

                return new Mesh(mesh);
            }

            /// loads meshes from .obj file
            public static NamedMeshList FromSceneObjFile(string path, bool combineAllObjects, ObjLoadSettings settings)
            {
                var mrSettings = new MRMeshLoadObjLoadSettings();
                mrSettings.customXf = settings.customXf;
                mrSettings.countSkippedFaces = settings.countSkippedFaces;
                mrSettings.callback = IntPtr.Zero;

                IntPtr errString = new IntPtr();

                IntPtr vector = mrMeshLoadFromSceneObjFile(path, combineAllObjects, ref mrSettings, ref errString);

                if (errString != IntPtr.Zero)
                {

                    var errData = mrStringData(errString);
                    string errorMessage = Marshal.PtrToStringAnsi(errData);
                    throw new SystemException(errorMessage);
                }

                int size = (int)mrVectorMeshLoadNamedMeshSize(vector);
                var meshes = new NamedMeshList(vector);

                for (int i = 0; i < size; i++)
                {
                    var mrNamedMesh = mrVectorMeshLoadNamedMeshGet(vector, (ulong)i);
                    var namedMesh = new NamedMesh();
                    namedMesh.name = Marshal.PtrToStringAnsi(mrNamedMesh.name);
                    namedMesh.mesh = new Mesh(mrNamedMesh.mesh);
                    namedMesh.mesh.SkipDisposingAtFinalize();
                    namedMesh.xf = new AffineXf3f(mrNamedMesh.xf);
                    namedMesh.skippedFaceCount = mrNamedMesh.skippedFaceCount;
                    namedMesh.duplicatedVertexCount = mrNamedMesh.duplicatedVertexCount;

                    meshes.Add(namedMesh);
                }

                return meshes;
            }
        }
    }
}
