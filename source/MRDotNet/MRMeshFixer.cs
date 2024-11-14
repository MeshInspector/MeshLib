using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    public class MeshFixer
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        unsafe private static extern IntPtr mrFindHoleComplicatingFaces(IntPtr mesh);

        public static BitSet FindHoleComplicatingFaces(Mesh mesh)
        {
            return new BitSet(mrFindHoleComplicatingFaces(mesh.mesh_));
        }
    }
}
