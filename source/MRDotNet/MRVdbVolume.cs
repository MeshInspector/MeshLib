using System;
using System.Collections.Generic;
using static MR.DotNet.Vector3i;
using static MR.DotNet.Vector3f;
using static MR.DotNet.VdbVolume;
using System.Runtime.InteropServices;

namespace MR
{
    public partial class DotNet
    {
        public class FloatGrid
        {
            internal IntPtr mrFloatGrid;
            internal FloatGrid(IntPtr mrFloatGrid)
            {
                this.mrFloatGrid = mrFloatGrid;
            }
        }
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

            internal MRVdbVolume mrVdbVolume_;

            internal VdbVolume(MRVdbVolume mrVdbVolume)
            {
                mrVdbVolume_ = mrVdbVolume;
            }
            public FloatGrid Data { get => new FloatGrid(mrVdbVolume_.data); }

            public Vector3i Dims { get => new Vector3i(mrVdbVolume_.dims); }
            public Vector3f VoxelSize { get => new Vector3f(mrVdbVolume_.voxelSize); }
            public float Min { get => mrVdbVolume_.min; }
            public float Max { get => mrVdbVolume_.max; }
        }

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
