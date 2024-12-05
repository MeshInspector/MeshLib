using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR
{
    public partial class DotNet
    {
        public enum SignDetectionMode
        {
            Unsigned,         ///< unsigned distance, useful for bidirectional `Shell` offset
            OpenVDB,          ///< sign detection from OpenVDB library, which is good and fast if input geometry is closed
            ProjectionNormal, ///< the sign is determined based on pseudonormal in closest mesh point (unsafe in case of self-intersections)
            WindingRule,      ///< ray intersection counter, significantly slower than ProjectionNormal and does not support holes in mesh
            HoleWindingRule   ///< computes winding number generalization with support of holes in mesh, slower than WindingRule
        };

        public struct OffsetParameters
        {
            /// Size of voxel in grid conversions;
            /// The user is responsible for setting some positive value here
            public float voxelSize = 0.0f;
            /// determines the method to compute distance sign
            public SignDetectionMode signDetectionMode = SignDetectionMode.OpenVDB;
            /// use FunctionVolume for voxel grid representation:
            ///  - memory consumption is approx. (z / (2 * thread_count)) less
            ///  - computation is about 2-3 times slower
            /// used only by \ref McOffsetMesh and \ref SharpOffsetMesh methods
            public bool memoryEfficient = false;
            public OffsetParameters() { }
        };

        public enum GeneralOffsetMode
        {
            /// create mesh using dual marching cubes from OpenVDB library
            Smooth,
            /// create mesh using standard marching cubes implemented in MeshLib
            Standard,
            /// create mesh using standard marching cubes with additional sharpening implemented in MeshLib
            Sharpening
        };

        public struct GeneralOffsetParameters
        {

            /// minimal surface deviation to introduce new vertex in a voxel, measured in voxelSize
            public float minNewVertDev = 1.0f / 25;
            /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes), measured in voxelSize
            public float maxNewRank2VertDev = 5;
            /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes), measured in voxelSize
            public float maxNewRank3VertDev = 2;
            /// correct positions of the input vertices using reference mesh by not more than this distance, measured in voxelSize;
            /// big correction can be wrong and result from self-intersections in the reference mesh
            public float maxOldVertPosCorrection = 0.5f;

            public GeneralOffsetMode mode = GeneralOffsetMode.Standard;

            public GeneralOffsetParameters() { }
        };

        public class Offset
        {
            [StructLayout(LayoutKind.Sequential)]
            internal struct MROffsetParameters
            {
                public float voxelSize = 0.0f;
                public IntPtr callBack = IntPtr.Zero;
                public SignDetectionMode signDetectionMode = SignDetectionMode.OpenVDB;
                public byte memoryEfficient = 0;
                public MROffsetParameters() { }
            };

            [StructLayout(LayoutKind.Sequential)]
            internal struct MRGeneralOffsetParameters
            {
                public float minNewVertDev = 1.0f / 25;
                public float maxNewRank2VertDev = 5;
                public float maxNewRank3VertDev = 2;
                public float maxOldVertPosCorrection = 0.5f;
                public GeneralOffsetMode mode = GeneralOffsetMode.Standard;

                public MRGeneralOffsetParameters() { }
            };

            ///
            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern float mrSuggestVoxelSize(MRMeshPart mp, float approxNumVoxels);


            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrOffsetMesh(MRMeshPart mp, float offset, ref MROffsetParameters parameters, ref IntPtr errorString);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrDoubleOffsetMesh(MRMeshPart mp, float offsetA, float offsetB, ref MROffsetParameters parameters, ref IntPtr errorString);


            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrMcOffsetMesh(MRMeshPart mp, float offset, ref MROffsetParameters parameters, ref IntPtr errorString);


            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrMcShellMeshRegion(IntPtr mesh, IntPtr region, float offset, ref MROffsetParameters parameters, ref IntPtr errorString);



            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrSharpOffsetMesh(MRMeshPart mp, float offset, ref MROffsetParameters parameters, ref MRGeneralOffsetParameters generalParams, ref IntPtr errorString);


            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrGeneralOffsetMesh(MRMeshPart mp, float offset, ref MROffsetParameters parameters, ref MRGeneralOffsetParameters generalParams, ref IntPtr errorString);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrThickenMesh(IntPtr mesh, float offset, ref MROffsetParameters parameters, ref MRGeneralOffsetParameters generalParams, ref IntPtr errorString);

            /// computes size of a cubical voxel to get approximately given number of voxels during rasterization
            public static float SuggestVoxelSize(MeshPart mp, float approxNumVoxels) => mrSuggestVoxelSize(mp.mrMeshPart, approxNumVoxels);

            /// Offsets mesh by converting it to distance field in voxels using OpenVDB library,
            /// signDetectionMode = Unsigned(from OpenVDB) | OpenVDB | HoleWindingRule,
            /// and then converts back using OpenVDB library (dual marching cubes),
            /// so result mesh is always closed
            /// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
            public static Mesh OffsetMesh(MeshPart mp, float offset, OffsetParameters parameters)
            {
                IntPtr errorStr = IntPtr.Zero;

                MROffsetParameters mrParameters = new MROffsetParameters();
                mrParameters.callBack = IntPtr.Zero;
                mrParameters.memoryEfficient = parameters.memoryEfficient ? (byte)1 : (byte)0;
                mrParameters.signDetectionMode = parameters.signDetectionMode;
                mrParameters.voxelSize = parameters.voxelSize;

                IntPtr res = mrOffsetMesh(mp.mrMeshPart, offset, ref mrParameters, ref errorStr);
                if (errorStr != IntPtr.Zero)
                {
                    string error = Marshal.PtrToStringAnsi(errorStr);
                    throw new Exception(error);
                }
                return new Mesh(res);
            }
            /// Offsets mesh by converting it to voxels and back two times
            /// only closed meshes allowed (only Offset mode)
            /// typically offsetA and offsetB have distinct signs
            /// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
            public static Mesh DoubleOffsetMesh(MeshPart mp, float offsetA, float offsetB, OffsetParameters parameters)
            {
                IntPtr errorStr = IntPtr.Zero;

                MROffsetParameters mrParameters = new MROffsetParameters();
                mrParameters.callBack = IntPtr.Zero;
                mrParameters.memoryEfficient = parameters.memoryEfficient ? (byte)1 : (byte)0;
                mrParameters.signDetectionMode = parameters.signDetectionMode;
                mrParameters.voxelSize = parameters.voxelSize;

                IntPtr res = mrDoubleOffsetMesh(mp.mrMeshPart, offsetA, offsetB, ref mrParameters, ref errorStr);
                if (errorStr != IntPtr.Zero)
                {
                    string error = Marshal.PtrToStringAnsi(errorStr);
                    throw new Exception(error);
                }
                return new Mesh(res);
            }
            /// Offsets mesh by converting it to distance field in voxels (using OpenVDB library if SignDetectionMode::OpenVDB or our implementation otherwise)
            /// and back using standard Marching Cubes, as opposed to Dual Marching Cubes in offsetMesh(...)
            /// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
            public static Mesh McOffsetMesh(MeshPart mp, float offset, OffsetParameters parameters)
            {
                IntPtr errorStr = IntPtr.Zero;

                MROffsetParameters mrParameters = new MROffsetParameters();
                mrParameters.callBack = IntPtr.Zero;
                mrParameters.memoryEfficient = parameters.memoryEfficient ? (byte)1 : (byte)0;
                mrParameters.signDetectionMode = parameters.signDetectionMode;
                mrParameters.voxelSize = parameters.voxelSize;

                IntPtr res = mrMcOffsetMesh(mp.mrMeshPart, offset, ref mrParameters, ref errorStr);
                if (errorStr != IntPtr.Zero)
                {
                    string error = Marshal.PtrToStringAnsi(errorStr);
                    throw new Exception(error);
                }
                return new Mesh(res);
            }
            /// Constructs a shell around selected mesh region with the properties that every point on the shall must
            ///  1. be located not further than given distance from selected mesh part,
            ///  2. be located not closer to not-selected mesh part than to selected mesh part.
            /// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
            public static Mesh McShellMeshRegion(MeshPart mp, float offset, OffsetParameters parameters)
            {
                IntPtr errorStr = IntPtr.Zero;

                MROffsetParameters mrParameters = new MROffsetParameters();
                mrParameters.callBack = IntPtr.Zero;
                mrParameters.memoryEfficient = parameters.memoryEfficient ? (byte)1 : (byte)0;
                mrParameters.signDetectionMode = parameters.signDetectionMode;
                mrParameters.voxelSize = parameters.voxelSize;

                if (mp.region is null)
                {
                    throw new Exception("region is null");
                }

                IntPtr res = mrMcShellMeshRegion(mp.mesh.mesh_, mp.region.bs_, offset, ref mrParameters, ref errorStr);
                if (errorStr != IntPtr.Zero)
                {
                    string error = Marshal.PtrToStringAnsi(errorStr);
                    throw new Exception(error);
                }
                return new Mesh(res);
            }
            /// Offsets mesh by converting it to voxels and back
            /// post process result using reference mesh to sharpen features
            /// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
            public static Mesh SharpOffsetMesh(MeshPart mp, float offset, OffsetParameters parameters, GeneralOffsetParameters generalParams)
            {
                IntPtr errorStr = IntPtr.Zero;

                MROffsetParameters mrParameters = new MROffsetParameters();
                mrParameters.callBack = IntPtr.Zero;
                mrParameters.memoryEfficient = parameters.memoryEfficient ? (byte)1 : (byte)0;
                mrParameters.signDetectionMode = parameters.signDetectionMode;
                mrParameters.voxelSize = parameters.voxelSize;

                MRGeneralOffsetParameters mrGeneralOffsetParameters = new MRGeneralOffsetParameters();
                mrGeneralOffsetParameters.maxNewRank2VertDev = generalParams.maxNewRank2VertDev;
                mrGeneralOffsetParameters.maxNewRank3VertDev = generalParams.maxNewRank3VertDev;
                mrGeneralOffsetParameters.maxOldVertPosCorrection = generalParams.maxOldVertPosCorrection;
                mrGeneralOffsetParameters.minNewVertDev = generalParams.minNewVertDev;
                mrGeneralOffsetParameters.mode = generalParams.mode;

                IntPtr res = mrSharpOffsetMesh(mp.mrMeshPart, offset, ref mrParameters, ref mrGeneralOffsetParameters, ref errorStr);
                if (errorStr != IntPtr.Zero)
                {
                    string error = Marshal.PtrToStringAnsi(errorStr);
                    throw new Exception(error);
                }
                return new Mesh(res);
            }
            /// Offsets mesh by converting it to voxels and back using one of three modes specified in the parameters
            /// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
            public static Mesh GeneralOffsetMesh(MeshPart mp, float offset, OffsetParameters parameters, GeneralOffsetParameters generalParams)
            {
                IntPtr errorStr = IntPtr.Zero;

                MROffsetParameters mrParameters = new MROffsetParameters();
                mrParameters.callBack = IntPtr.Zero;
                mrParameters.memoryEfficient = parameters.memoryEfficient ? (byte)1 : (byte)0;
                mrParameters.signDetectionMode = parameters.signDetectionMode;
                mrParameters.voxelSize = parameters.voxelSize;

                MRGeneralOffsetParameters mrGeneralOffsetParameters = new MRGeneralOffsetParameters();
                mrGeneralOffsetParameters.maxNewRank2VertDev = generalParams.maxNewRank2VertDev;
                mrGeneralOffsetParameters.maxNewRank3VertDev = generalParams.maxNewRank3VertDev;
                mrGeneralOffsetParameters.maxOldVertPosCorrection = generalParams.maxOldVertPosCorrection;
                mrGeneralOffsetParameters.minNewVertDev = generalParams.minNewVertDev;
                mrGeneralOffsetParameters.mode = generalParams.mode;

                IntPtr res = mrGeneralOffsetMesh(mp.mrMeshPart, offset, ref mrParameters, ref mrGeneralOffsetParameters, ref errorStr);
                if (errorStr != IntPtr.Zero)
                {
                    string error = Marshal.PtrToStringAnsi(errorStr);
                    throw new Exception(error);
                }
                return new Mesh(res);
            }
            /// in case of positive offset, returns the mesh consisting of offset mesh merged with inversed original mesh (thickening mode);
            /// in case of negative offset, returns the mesh consisting of inversed offset mesh merged with original mesh (hollowing mode);
            /// if your input mesh is open then please specify params.signDetectionMode = SignDetectionMode::Unsigned, and you will get open mesh (with several components) on output
            /// if your input mesh is closed then please specify another sign detection mode, and you will get closed mesh (with several components) on output;
            /// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr

            public static Mesh ThickenMesh(Mesh mesh, float offset, OffsetParameters parameters, GeneralOffsetParameters generalParams)
            {
                IntPtr errorStr = IntPtr.Zero;

                MROffsetParameters mrParameters = new MROffsetParameters();
                mrParameters.callBack = IntPtr.Zero;
                mrParameters.memoryEfficient = parameters.memoryEfficient ? (byte)1 : (byte)0;
                mrParameters.signDetectionMode = parameters.signDetectionMode;
                mrParameters.voxelSize = parameters.voxelSize;

                MRGeneralOffsetParameters mrGeneralOffsetParameters = new MRGeneralOffsetParameters();
                mrGeneralOffsetParameters.maxNewRank2VertDev = generalParams.maxNewRank2VertDev;
                mrGeneralOffsetParameters.maxNewRank3VertDev = generalParams.maxNewRank3VertDev;
                mrGeneralOffsetParameters.maxOldVertPosCorrection = generalParams.maxOldVertPosCorrection;
                mrGeneralOffsetParameters.minNewVertDev = generalParams.minNewVertDev;
                mrGeneralOffsetParameters.mode = generalParams.mode;

                IntPtr res = mrThickenMesh(mesh.mesh_, offset, ref mrParameters, ref mrGeneralOffsetParameters, ref errorStr);
                if (errorStr != IntPtr.Zero)
                {
                    string error = Marshal.PtrToStringAnsi(errorStr);
                    throw new Exception(error);
                }
                return new Mesh(res);
            }
        }
    }
}
