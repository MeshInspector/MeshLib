using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace MR
{
    public partial class DotNet
    {
        public class PointsLoad
        {
            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrPointsLoadFromAnySupportedFormat(string filename, IntPtr settings, ref IntPtr errorString);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern void mrLoadIOExtras();

            /// loads point cloud from file of any supported format
            public static PointCloud FromAnySupportedFormat(string path)
            {
                mrLoadIOExtras();

                IntPtr errString = IntPtr.Zero;
                var mesh = mrPointsLoadFromAnySupportedFormat(path, IntPtr.Zero, ref errString);

                if (errString != IntPtr.Zero)
                {
                    var errData = mrStringData(errString);
                    string errorMessage = Marshal.PtrToStringAnsi(errData);
                    throw new SystemException(errorMessage);
                }

                return new PointCloud(mesh);
            }
        }
    }
}