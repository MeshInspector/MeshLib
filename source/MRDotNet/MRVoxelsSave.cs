using System;
using System.Runtime.InteropServices;
using static MR.DotNet.VdbVolume;

namespace MR
{
    public partial class DotNet
    {
        public class VoxelsSave
        {
            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern void mrVoxelsSaveToAnySupportedFormat( ref MRVdbVolume volume, string file, IntPtr cb, ref IntPtr errorStr );


            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrStringData(IntPtr str);
            
            /// Saves voxels in a file, detecting the format from file extension
            public static void ToAnySupportedFormat( VdbVolume volume, string path)
            {
                IntPtr errorStr = IntPtr.Zero;
                var nativeVolume = volume.volume();
                mrVoxelsSaveToAnySupportedFormat( ref nativeVolume, path, IntPtr.Zero, ref errorStr);
                if (errorStr != IntPtr.Zero)
                {
                    var errData = mrStringData(errorStr);
                    string errorMessage = Marshal.PtrToStringAnsi(errData);
                    throw new SystemException(errorMessage);
                }
            }
        }
    }
}