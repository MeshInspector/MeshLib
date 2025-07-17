using System;
using System.Runtime.InteropServices;
using static MR.DotNet.VdbVolume;
using static MR.DotNet.Vector3f;

namespace MR
{
    public partial class DotNet
    {
        // Conversion type
        public enum MeshToVolumeSettingsType
        {
            Signed,  // only closed meshes can be converted with signed type
            Unsigned // this type leads to shell like iso-surfaces
        }
        // Parameters structure for meshToVolume function
        public struct MeshToVolumeSettings
        {
            // Conversion type
            public MeshToVolumeSettingsType type = MeshToVolumeSettingsType.Unsigned;
            // the number of voxels around surface to calculate distance in (should be positive)
            public float surfaceOffset = 3.0f;
            public Vector3f voxelSize = Vector3f.Diagonal(1.0f);
            // mesh initial transform
            public AffineXf3f worldXf = new AffineXf3f();
            // optional output: xf to original mesh (respecting worldXf)
            public AffineXf3f? outXf = null;

            public MeshToVolumeSettings() { }
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRMeshToVolumeSettings
        {
            public MeshToVolumeSettingsType type = MeshToVolumeSettingsType.Unsigned;
            public float surfaceOffset = 3.0f;
            public MRVector3f voxelSize = new MRVector3f();
            public MRAffineXf3f worldXf = new MRAffineXf3f();
            public IntPtr outXf = IntPtr.Zero;
            public IntPtr cb = IntPtr.Zero;

            public MRMeshToVolumeSettings() { }
        }

        /// parameters of FloatGrid to Mesh conversion using Dual Marching Cubes algorithm
        public struct GridToMeshSettings
        {
            /// the size of each voxel in the grid
            public Vector3f voxelSize = Vector3f.Diagonal(0.0f);
            /// layer of grid with this value would be converted in mesh; isoValue can be negative only in level set grids
            public float isoValue = 0.0f;
            /// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones (curvature can be lost on high values)
            public float adaptivity = 0.0f;
            /// if the mesh exceeds this number of faces, an error returns
            public int maxFaces = int.MaxValue;
            /// if the mesh exceeds this number of vertices, an error returns
            public int maxVertices = int.MaxValue;
            public bool relaxDisorientedTriangles = true;

            public GridToMeshSettings() { }
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRGridToMeshSettings
        {
            public MRVector3f voxelSize = new MRVector3f();
            public float isoValue = 0.0f;
            public float adaptivity = 0.0f;
            public int maxFaces = int.MaxValue;
            public int maxVertices = int.MaxValue;
            public byte relaxDisorientedTriangles = 1;
            public IntPtr cb = IntPtr.Zero;

            public MRGridToMeshSettings() { }
        }

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        unsafe private static extern void mrVdbConversionsEvalGridMinMax( IntPtr grid, float* min, float* max );

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        private static extern MRVdbVolume mrVdbConversionsMeshToVolume( IntPtr mesh, ref MRMeshToVolumeSettings settings, ref IntPtr errorStr );

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        private static extern MRVdbVolume mrVdbConversionsFloatGridToVdbVolume( IntPtr grid );

        [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrVdbConversionsGridToMesh( IntPtr grid, ref MRGridToMeshSettings settings, ref IntPtr errorStr );

        // eval min max value from FloatGrid
        unsafe public static void EvalGridMinMax( FloatGrid grid, out float min, out float max )
        {
            fixed ( float* minP = &min, maxP = &max ) mrVdbConversionsEvalGridMinMax( grid.mrFloatGrid, minP, maxP );
        }

        // convert mesh to volume in (0,0,0)-(dim.x,dim.y,dim.z) grid box
        public static VdbVolume MeshToVolume( Mesh mesh, MeshToVolumeSettings settings )
        {
            IntPtr errorStr = IntPtr.Zero;
            MRMeshToVolumeSettings mrSettings = new MRMeshToVolumeSettings();
            mrSettings.type = settings.type;
            mrSettings.surfaceOffset = settings.surfaceOffset;
            mrSettings.voxelSize = settings.voxelSize.vec_;
            mrSettings.worldXf = settings.worldXf.xf_;
            if ( settings.outXf != null )
            {
                mrSettings.outXf = settings.outXf.XfAddr();
            }

            var mrVolume = mrVdbConversionsMeshToVolume( mesh.mesh_, ref mrSettings, ref errorStr );

            if ( errorStr != IntPtr.Zero )
            {
                var errData = mrStringData( errorStr );
                string errorMessage = MarshalNativeUtf8ToManagedString( errData );
                throw new SystemException( errorMessage );
            }

            return new VdbVolume( mrVolume );
        }

        // fills VdbVolume data from FloatGrid (does not fill voxels size, cause we expect it outside)
        public static VdbVolume FloatGridToVdbVolume( FloatGrid grid )
        {
            return new VdbVolume( mrVdbConversionsFloatGridToVdbVolume( grid.mrFloatGrid ) );
        }
        
        /// converts OpenVDB Grid into mesh using Dual Marching Cubes algorithm
        public static Mesh GridToMesh( FloatGrid grid, GridToMeshSettings settings )
        {
            IntPtr errorStr = IntPtr.Zero;
            
            MRGridToMeshSettings mrSettings = new MRGridToMeshSettings();
            mrSettings.voxelSize = settings.voxelSize.vec_;
            mrSettings.isoValue = settings.isoValue;
            mrSettings.adaptivity = settings.adaptivity;
            mrSettings.maxFaces = settings.maxFaces;
            mrSettings.maxVertices = settings.maxVertices;
            mrSettings.relaxDisorientedTriangles = settings.relaxDisorientedTriangles ? (byte)1 : (byte)0;

            var mrMesh = mrVdbConversionsGridToMesh( grid.mrFloatGrid, ref mrSettings, ref errorStr );
            if ( errorStr != IntPtr.Zero )
            {
                var errData = mrStringData( errorStr );
                string errorMessage = MarshalNativeUtf8ToManagedString( errData );
                throw new SystemException( errorMessage );
            }
            return new Mesh( mrMesh );
        }
    }
}
