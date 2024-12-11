using System;
using System.Runtime.InteropServices;
using static MR.DotNet.VdbVolume;

namespace MR
{
    public partial class DotNet
    {
        public class VoxelsSave
        {
            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern void mrVoxelsSaveToAnySupportedFormat( ref MRVdbVolume volume, string file, IntPtr cb, ref IntPtr errorStr );


            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrStringData(IntPtr str);

            public static void ToAnySupportedFormat( VdbVolume volume, string path)
            {
                IntPtr errorStr = IntPtr.Zero;
                mrVoxelsSaveToAnySupportedFormat( ref volume.mrVdbVolume_, path, IntPtr.Zero, ref errorStr);
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