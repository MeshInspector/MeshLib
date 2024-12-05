using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using static MR.DotNet.AffineXf3f;

namespace MR
{
    public partial class DotNet
    {
        public struct NamedMeshXf
        {
            public string name = "";
            public AffineXf3f toWorld = new AffineXf3f();
            public Mesh? mesh = null;
            public NamedMeshXf() { }
        }

        [StructLayout(LayoutKind.Sequential)]
        struct MRMeshSaveNamedXfMesh
        {
            public string name = "";
            public MRAffineXf3f toWorld = new MRAffineXf3f();
            public IntPtr mesh = IntPtr.Zero;

            public MRMeshSaveNamedXfMesh() { }
        }

        public class MeshSave
        {
            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrStringData(IntPtr str);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern void mrMeshSaveSceneToObj(IntPtr objects, ulong objectsNum, string file, ref IntPtr errorString);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern void mrLoadIOExtras();

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            unsafe private static extern void mrMeshSaveToAnySupportedFormat(IntPtr mesh, string file, IntPtr* errorStr);

            /// saves mesh to file of any supported format
            unsafe public static void ToAnySupportedFormat(Mesh mesh, string path)
            {
                mrLoadIOExtras();

                IntPtr errString = new IntPtr();
                mrMeshSaveToAnySupportedFormat(mesh.mesh_, path, &errString);
                if (errString != IntPtr.Zero)
                {
                    var errData = mrStringData(errString);
                    string errorMessage = Marshal.PtrToStringAnsi(errData);
                    throw new SystemException(errorMessage);
                }
            }

            /// saves a number of named meshes in .obj file
            public static void SceneToObj(List<NamedMeshXf> meshes, string file)
            {
                int sizeOfNamedXfMesh = Marshal.SizeOf(typeof(MRMeshSaveNamedXfMesh));
                IntPtr nativeMeshes = Marshal.AllocHGlobal(meshes.Count * sizeOfNamedXfMesh);

                try
                {
                    for (int i = 0; i < meshes.Count; i++)
                    {
                        MRMeshSaveNamedXfMesh mrMesh = new MRMeshSaveNamedXfMesh();
                        mrMesh.name = meshes[i].name;
                        mrMesh.toWorld = meshes[i].toWorld.xf_;
                        var mesh = meshes[i].mesh;
                        if (mesh != null)
                            mrMesh.mesh = mesh.mesh_;

                        Marshal.StructureToPtr(mrMesh, IntPtr.Add(nativeMeshes, i * sizeOfNamedXfMesh), false);
                    }

                    IntPtr errString = new IntPtr();
                    mrMeshSaveSceneToObj(nativeMeshes, (ulong)meshes.Count, file, ref errString);
                }
                finally
                {
                    Marshal.FreeHGlobal(nativeMeshes);
                }
            }
        }


    };
}
