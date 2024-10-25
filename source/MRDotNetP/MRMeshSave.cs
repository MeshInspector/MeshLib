using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using static MR.DotNet.AffineXf3f;

namespace MR.DotNet
{
    public struct NamedMeshXf
    {
        public string name;
        public AffineXf3f toWorld;
        public Mesh mesh;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct MRMeshSaveNamedXfMesh
    {
        public string name;
        public MRAffineXf3f toWorld;
        public IntPtr mesh;
    }

    public class MeshSave
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrStringData(IntPtr str);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshSaveSceneToObj( IntPtr objects, ulong objectsNum, string file, ref IntPtr errorString );
        /// saves a number of named meshes in .obj file
        public static void SceneToObj(List<NamedMeshXf> meshes, string file )
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
                    mrMesh.mesh = meshes[i].mesh.mesh_;

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
