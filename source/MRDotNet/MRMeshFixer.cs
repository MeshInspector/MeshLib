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
        /// returns all faces that complicate one of mesh holes;
        /// hole is complicated if it passes via one vertex more than once;
        /// deleting such faces simplifies the holes and makes them easier to fill
        public static BitSet FindHoleComplicatingFaces(Mesh mesh)
        {
            return new BitSet(mrFindHoleComplicatingFaces(mesh.mesh_));
        }
    }
}
