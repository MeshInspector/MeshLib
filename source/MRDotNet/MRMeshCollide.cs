using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using static MR.DotNet;

namespace MR
{
    public partial class DotNet
    {

        [DllImport("MRMeshC", CharSet = CharSet.Auto)]
        private static extern bool mrIsInside(ref MRMeshPart a, ref MRMeshPart b, IntPtr rigidB2A);

        /**
        * \brief checks that arbitrary mesh part A is inside of closed mesh part B
        * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
        */
        public static bool IsInside(MeshPart meshA, MeshPart meshB, AffineXf3f? rigidB2A = null)
        {
            return mrIsInside(ref meshA.mrMeshPart, ref meshB.mrMeshPart, rigidB2A?.XfAddr() ?? IntPtr.Zero);
        }
    }
}
