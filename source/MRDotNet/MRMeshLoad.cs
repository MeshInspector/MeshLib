using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using static MR.DotNet.AffineXf3f;

namespace MR.DotNet
{
    public struct ObjLoadSettings
    {
        /// if true then vertices will be returned relative to some transformation to avoid precision loss
        public bool customXf;
        /// if true, the number of skipped faces (faces than can't be created) will be counted
        public bool countSkippedFaces;
    };

    public struct NamedMesh
    {
        public string name;
        public Mesh mesh;
        /// transform of the loaded mesh, not identity only if ObjLoadSettings.customXf
        public AffineXf3f xf;
        /// counter of skipped faces (faces than can't be created), not zero only if ObjLoadSettings.countSkippedFaces
        public int skippedFaceCount;
        /// counter of duplicated vertices (that created for resolve non-manifold geometry)
        public int duplicatedVertexCount;
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRMeshLoadObjLoadSettings
    {
        /// if true then vertices will be returned relative to some transformation to avoid precision loss
        public bool customXf;
        /// if true, the number of skipped faces (faces than can't be created) will be counted
        public bool countSkippedFaces;
        /// callback for set progress and stop process
        public IntPtr callback;
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRMeshLoadNamedMesh
    {
        public IntPtr name;
        public IntPtr mesh;
        /// transform of the loaded mesh, not identity only if ObjLoadSettings.customXf
        public MRAffineXf3f xf;
        /// counter of skipped faces (faces than can't be created), not zero only if ObjLoadSettings.countSkippedFaces
        public int skippedFaceCount;
        /// counter of duplicated vertices (that created for resolve non-manifold geometry)
        public int duplicatedVertexCount;
    };

    public class MeshLoad
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRMeshLoadNamedMesh mrVectorMeshLoadNamedMeshGet( IntPtr vector, ulong index );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ulong mrVectorMeshLoadNamedMeshSize( IntPtr vector );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrVectorMeshLoadNamedMeshFree(IntPtr vector);


        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshLoadFromSceneObjFile( string file, bool combineAllObjects, ref MRMeshLoadObjLoadSettings settings, ref IntPtr errorString );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrStringData(IntPtr str);

        /// loads meshes from .obj file
        public static List<NamedMesh> FromSceneObjFile(string path, bool combineAllObjects, ObjLoadSettings settings)
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
            List<NamedMesh> meshes = new List<NamedMesh>(size);

            for (int i = 0; i < size; i++)
            {
                var mrNamedMesh = mrVectorMeshLoadNamedMeshGet(vector, (ulong)i);
                var namedMesh = new NamedMesh();
                namedMesh.name = Marshal.PtrToStringAnsi(mrNamedMesh.name);
                namedMesh.mesh = new Mesh(mrNamedMesh.mesh);
                namedMesh.xf = new AffineXf3f(mrNamedMesh.xf);
                namedMesh.skippedFaceCount = mrNamedMesh.skippedFaceCount;
                namedMesh.duplicatedVertexCount = mrNamedMesh.duplicatedVertexCount;

                meshes.Add(namedMesh);
            }

            return meshes;
        }
    }
}
