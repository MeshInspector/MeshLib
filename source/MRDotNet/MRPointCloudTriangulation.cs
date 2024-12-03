using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;

namespace MR
{
    public partial class DotNet
    {

        /**
        * \brief Parameters of point cloud triangulation
        */
        public struct TriangulationParameters
        {
            /**
             * \brief The number of nearest neighbor points to use for building of local triangulation
             * \note Too small value can make not optimal triangulation and additional holes
             * Too big value increases difficulty of optimization and decreases performance    
             */
            public int numNeighbours = 16;
            /**
             * Radius of neighborhood around each point to consider for building local triangulation.
             * This is an alternative to numNeighbours parameter.
             * Please set to positive value only one of them.
             */
            public float radius = 0;
            /**
             * \brief Critical angle of triangles in local triangulation (angle between triangles in fan should be less then this value)    
             */
            public float critAngle = (float)Math.PI / 2;

            /// the vertex is considered as boundary if its neighbor ring has angle more than this value
            public float boundaryAngle = 0.9f * (float)Math.PI;
            /**
             * \brief Critical length of hole (all holes with length less then this value will be filled)
             * \details If value is subzero it is set automaticly to 0.7*bbox.diagonal()
             */
            public float critHoleLength = float.MinValue;

            /// automatic increase of the radius if points outside can make triangles from original radius not-Delone
            public bool automaticRadiusIncrease = true;

            /// optional: if provided this cloud will be used for searching of neighbors (so it must have same validPoints)
            public PointCloud? searchNeighbors = null;

            public TriangulationParameters() { }
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRTriangulationParameters
        {
            public int numNeighbours = 16;
            public float radius = 0;
            public float critAngle = (float)Math.PI / 2;
            public float boundaryAngle = 0.9f * (float)Math.PI;
            public float critHoleLength = float.MinValue;
            public byte automaticRadiusIncrease = 1;
            public IntPtr searchNeighbors = IntPtr.Zero;
            public MRTriangulationParameters() { }
        };

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrTriangulatePointCloud(IntPtr pointCloud, ref MRTriangulationParameters parameters);

        /**        
        * \brief Creates mesh from given point cloud according params
        * Returns empty optional if was interrupted by progress bar
        */
        public static Mesh? TriangulatePointCloud(PointCloud pc, TriangulationParameters parameters)
        {
            var mrParameters = new MRTriangulationParameters();
            mrParameters.numNeighbours = parameters.numNeighbours;
            mrParameters.radius = parameters.radius;
            mrParameters.critAngle = parameters.critAngle;
            mrParameters.boundaryAngle = parameters.boundaryAngle;
            mrParameters.critHoleLength = parameters.critHoleLength;
            mrParameters.automaticRadiusIncrease = parameters.automaticRadiusIncrease ? (byte)1 : (byte)0;
            mrParameters.searchNeighbors = parameters.searchNeighbors?.pc_ ?? IntPtr.Zero;

            return new Mesh(mrTriangulatePointCloud(pc.pc_, ref mrParameters));
        }
    }
}
