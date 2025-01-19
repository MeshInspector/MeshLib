using System;
using System.Collections.Generic;
using static MR.DotNet.Vector3i;
using static MR.DotNet.Vector3f;
using static MR.DotNet.VdbVolume;
using System.Runtime.InteropServices;
using static MR.DotNet.Box3f;
using static MR.DotNet.Box3i;

namespace MR
{
    public partial class DotNet
    {        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFloatGridResampledUniformly( IntPtr grid, float voxelScale, IntPtr cb );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFloatGridResampled( IntPtr grid, ref MRVector3f voxelScale, IntPtr cb );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFloatGridCropped( IntPtr grid, ref MRBox3i box, IntPtr cb );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern float mrFloatGridGetValue( IntPtr grid, ref MRVector3i p );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrFloatGridSetValue(IntPtr grid, ref MRVector3i p, float value );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrFloatGridSetValueForRegion(IntPtr grid, IntPtr region, float value );

        /// stores a pointer to a native OpenVDB object
        public class FloatGrid
        {
            internal IntPtr mrFloatGrid;
            internal FloatGrid(IntPtr mrFloatGrid)
            {
                this.mrFloatGrid = mrFloatGrid;
            }
        }
        
        /// resample this grid with voxel size uniformly scaled by voxelScale
        public static FloatGrid Resampled(FloatGrid grid, float voxelScale)
        {
            return new FloatGrid(mrFloatGridResampledUniformly(grid.mrFloatGrid, voxelScale, IntPtr.Zero));
        }

        /// resample this grid with voxel size scaled by voxelScale in each dimension
        public static FloatGrid Resampled(FloatGrid grid, Vector3f voxelScale)
        {
            return new FloatGrid(mrFloatGridResampled(grid.mrFloatGrid, ref voxelScale.vec_, IntPtr.Zero));
        }

        /// returns cropped grid
        public static FloatGrid Cropped(FloatGrid grid, Box3i box)
        {
            return new FloatGrid(mrFloatGridCropped(grid.mrFloatGrid, ref box.boxRef(), IntPtr.Zero));
        }

        /// returns the value at given voxel
        public static float GetValue(FloatGrid grid, Vector3i p) => mrFloatGridGetValue(grid.mrFloatGrid, ref p.vec_);

        /// sets given voxel
        public static void SetValue(FloatGrid grid, Vector3i p, float value) => mrFloatGridSetValue(grid.mrFloatGrid, ref p.vec_, value);
        
        /// sets given region voxels value
        /// \note region is in grid space (0 voxel id is minimum active voxel in grid)
        public static void SetValue(FloatGrid grid, VoxelBitSet region, float value) => mrFloatGridSetValueForRegion(grid.mrFloatGrid, region.bs_, value);

        /// represents a box in 3D space subdivided on voxels stored in data;
        /// and stores minimum and maximum values among all valid voxels
        public class VdbVolume
        {
            [StructLayout(LayoutKind.Sequential)]
            internal struct MRVdbVolume
            {
                public IntPtr data = IntPtr.Zero;
                public MRVector3i dims = new MRVector3i();
                public MRVector3f voxelSize = new MRVector3f();
                public float min = 0.0f;
                public float max = 0.0f;
                public MRVdbVolume() { }
            }

            private MRVdbVolume mrVdbVolume_;
            private Vector3f voxelSize_;

            internal VdbVolume(MRVdbVolume mrVdbVolume)
            {
                mrVdbVolume_ = mrVdbVolume;
                voxelSize_ = new Vector3f(mrVdbVolume_.voxelSize);
            }

            internal MRVdbVolume volume()
            {
                mrVdbVolume_.voxelSize = voxelSize_.vec_;
                return mrVdbVolume_;
            }

            /// returns the pointer to the data
            public FloatGrid Data { get => new FloatGrid(mrVdbVolume_.data); }
            /// returns the dimensions of the volume            
            public Vector3i Dims { get => new Vector3i(mrVdbVolume_.dims); }
            /// returns the size of voxel
            public Vector3f VoxelSize { get => voxelSize_; set => voxelSize_ = value; }
            public float Min { get => mrVdbVolume_.min; }
            public float Max { get => mrVdbVolume_.max; }
        }
        /// stores a list of volumes
        public class VdbVolumes : List<VdbVolume>, IDisposable 
        {
            /// gets the volumes' value at index
            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern MRVdbVolume mrVdbVolumesGet( IntPtr volumes, ulong index );

            /// gets the volumes' size
            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern ulong mrVdbVolumesSize( IntPtr volumes );

            /// deallocates the VdbVolumes object
            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern void mrVdbVolumesFree(IntPtr volumes);

            internal IntPtr mrVdbVolumes_;
            private bool disposed_ = false;
            internal VdbVolumes(IntPtr mrVdbVolumes)
            : base((int)mrVdbVolumesSize(mrVdbVolumes))
            {
                mrVdbVolumes_ = mrVdbVolumes;

                for (int i = 0; i < this.Capacity; i++)
                {
                    this.Add(new VdbVolume(mrVdbVolumesGet(mrVdbVolumes_, (ulong)i)));
                }
            }
            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            protected virtual void Dispose(bool disposing)
            {
                if (!disposed_)
                {
                    if (mrVdbVolumes_ != IntPtr.Zero)
                    {
                        mrVdbVolumesFree(mrVdbVolumes_);
                        mrVdbVolumes_ = IntPtr.Zero;
                    }

                    disposed_ = true;
                }
            }

            ~VdbVolumes()
            {
                Dispose(false);
            }
        }
    };
}
