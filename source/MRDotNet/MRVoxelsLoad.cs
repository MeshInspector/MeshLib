using System;
using System.Runtime.InteropServices;

namespace MR
{
    public partial class DotNet
    {
        public class VoxelsLoad
        {
            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrVoxelsLoadFromAnySupportedFormat( string file, IntPtr cb, ref IntPtr errorStr );

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrStringData(IntPtr str);

            public static VdbVolumes FromAnySupportedFormat( string path )
            {
                IntPtr errorStr = IntPtr.Zero;
                IntPtr res = mrVoxelsLoadFromAnySupportedFormat( path, IntPtr.Zero, ref errorStr );
                if ( errorStr != IntPtr.Zero )
                {
                    var errData = mrStringData( errorStr );
                    string errorMessage = Marshal.PtrToStringAnsi( errData );
                    throw new SystemException( errorMessage );
                }
                return new VdbVolumes( res );
            }
        }
    }
}
