
using System;
using System.Runtime.InteropServices;
using static MR.DotNet.SelfIntersections;

namespace MR
{
    public partial class DotNet
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRRelaxParams
        {
            public int iterations = 1;
            public IntPtr region = IntPtr.Zero;
            public float force = 0.5f;
            public byte limitNearInitial = 0;
            public float maxInitialDist = 0f;

            public MRRelaxParams() { }
        };

        /// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
        /// \return true if was finished successfully, false if was interrupted by progress callback
        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool mrRelax(IntPtr mesh, ref MRRelaxParams settings, IntPtr cb);

        /// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified ) \n
        /// do not really keeps volume but tries hard
        /// \return true if the operation completed successfully, and false if it was interrupted by the progress callback.
        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool mrRelaxKeepVolume(IntPtr mesh, ref MRRelaxParams settings, IntPtr cb);

        public struct RelaxParams
        {
            /// number of iterations
            public int iterations = 1;

            /// region to relax
            public VertBitSet? region = null;

            /// speed of relaxing, typical values (0.0, 0.5]
            public float force = 0.5f;

            /// if true then maximal displacement of each point during denoising will be limited
            public bool limitNearInitial = false;

            /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
            public float maxInitialDist = 0f;

            public RelaxParams() { }
        }

        /// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
        public static bool Relax(ref Mesh mesh, RelaxParams relaxParams )
        {
            MRRelaxParams mrParameters = new MRRelaxParams();
            mrParameters.iterations = relaxParams.iterations;
            mrParameters.region = relaxParams.region is null ? (IntPtr)null : relaxParams.region.bs_;
            mrParameters.force = relaxParams.force;
            mrParameters.limitNearInitial = relaxParams.limitNearInitial ? (byte)1 : (byte)0;
            mrParameters.maxInitialDist = relaxParams.maxInitialDist;

            return mrRelax(mesh.mesh_, ref mrParameters, IntPtr.Zero);
        }

        /// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified ) \n
        /// do not really keeps volume but tries hard
        public static bool RelaxKeepVolume(ref Mesh mesh, RelaxParams relaxParams)
        {
            MRRelaxParams mrParameters = new MRRelaxParams();
            mrParameters.iterations = relaxParams.iterations;
            mrParameters.region = relaxParams.region is null ? (IntPtr)null : relaxParams.region.bs_;
            mrParameters.force = relaxParams.force;
            mrParameters.limitNearInitial = relaxParams.limitNearInitial ? (byte)1 : (byte)0;
            mrParameters.maxInitialDist = relaxParams.maxInitialDist;

            return mrRelaxKeepVolume(mesh.mesh_, ref mrParameters, IntPtr.Zero);
        }
    }
}
